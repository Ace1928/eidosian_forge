import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
@logs_state_cli_group.command(name='job')
@click.option('--id', 'submission_id', required=True, type=str, help='Retrieves the logs from a submission job with submission id,i.e. raysubmit_XXX')
@address_option
@log_follow_option
@log_tail_option
@log_interval_option
@log_timeout_option
@click.pass_context
@PublicAPI(stability='stable')
def log_job(ctx, submission_id: Optional[str], address: Optional[str], follow: bool, tail: int, interval: float, timeout: int):
    """Get logs associated with a submission job.

    Example:

        Follow the log file from a submission job with submission id raysumbit_xxx.

        ```
        ray logs job --id raysubmit_xxx
        ```

        Follow the submission job log.

        ```
        ray logs jobs --id raysubmit_xxx --follow

        ```

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI is failed to query the data.
        MissingParameter if inputs are missing.
    """
    _print_log(address=address, tail=tail, follow=follow, interval=interval, timeout=timeout, submission_id=submission_id)