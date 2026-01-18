import json
import os
import sys
import pprint
import time
from subprocess import list2cmdline
from typing import Optional, Tuple, Union, Dict, Any
import click
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
from ray._private.utils import get_or_create_event_loop
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args
from ray.job_submission import JobStatus, JobSubmissionClient
from ray.dashboard.modules.job.cli_utils import add_common_job_options
from ray.dashboard.modules.job.utils import redact_url_password
from ray.util.annotations import PublicAPI
from ray._private.utils import parse_resources_json, parse_metadata_json
@job_cli_group.command()
@click.option('--address', type=str, default=None, required=False, help='Address of the Ray cluster to connect to. Can also be specified using the RAY_ADDRESS environment variable.')
@click.argument('job-id', type=str)
@click.option('-f', '--follow', is_flag=True, type=bool, default=False, help='If set, follow the logs (like `tail -f`).')
@add_common_job_options
@add_click_logging_options
@PublicAPI(stability='stable')
def logs(address: Optional[str], job_id: str, follow: bool, headers: Optional[str], verify: Union[bool, str]):
    """Gets the logs of a job.

    Example:
        `ray job logs <my_job_id>`
    """
    client = _get_sdk_client(address, headers=headers, verify=verify)
    sdk_version = client.get_version()
    if follow:
        if int(sdk_version) > 0:
            get_or_create_event_loop().run_until_complete(_tail_logs(client, job_id))
        else:
            cli_logger.warning(f'Tailing logs is not enabled for the Jobs SDK client version {sdk_version}. Please upgrade Ray to latest version for this feature.')
    else:
        cli_logger.print(client.get_job_logs(job_id), end='', no_format=True)