import copy
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import click
import yaml
import ray
from ray._private.usage import usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.cluster_dump import (
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.providers import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.experimental.internal_kv import _internal_kv_put, internal_kv_get_gcs_client
from ray.util.debug import log_once
def get_cluster_dump_archive(cluster_config_file: Optional[str]=None, host: Optional[str]=None, ssh_user: Optional[str]=None, ssh_key: Optional[str]=None, docker: Optional[str]=None, local: Optional[bool]=None, output: Optional[str]=None, logs: bool=True, debug_state: bool=True, pip: bool=True, processes: bool=True, processes_verbose: bool=False, tempfile: Optional[str]=None) -> Optional[str]:
    content_str = ''
    if logs:
        content_str += '  - The logfiles of your Ray session\n    This usually includes Python outputs (stdout/stderr)\n'
    if debug_state:
        content_str += '  - Debug state information on your Ray cluster \n    e.g. number of workers, drivers, objects, etc.\n'
    if pip:
        content_str += '  - Your installed Python packages (`pip freeze`)\n'
    if processes:
        content_str += '  - Information on your running Ray processes\n    This includes command line arguments\n'
    cli_logger.warning(f'You are about to create a cluster dump. This will collect data from cluster nodes.\n\nThe dump will contain this information:\n\n{content_str}\nIf you are concerned about leaking private information, extract the archive and inspect its contents before sharing it with anyone.')
    cluster_config_file, hosts, ssh_user, ssh_key, docker, cluster_name = _info_from_params(cluster_config_file, host, ssh_user, ssh_key, docker)
    nodes = [Node(host=h, ssh_user=ssh_user, ssh_key=ssh_key, docker_container=docker) for h in hosts]
    if not nodes:
        cli_logger.error('No nodes found. Specify with `--host` or by passing a ray cluster config to `--cluster`.')
        return None
    if cluster_config_file:
        nodes[0].is_head = True
    if local is None:
        local = not bool(cluster_config_file)
    parameters = GetParameters(logs=logs, debug_state=debug_state, pip=pip, processes=processes, processes_verbose=processes_verbose)
    with Archive(file=tempfile) as archive:
        if local:
            create_archive_for_local_and_remote_nodes(archive, remote_nodes=nodes, parameters=parameters)
        else:
            create_archive_for_remote_nodes(archive, remote_nodes=nodes, parameters=parameters)
    if not output:
        if cluster_name:
            filename = f'{cluster_name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.tar.gz'
        else:
            filename = f'collected_logs_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.tar.gz'
        output = os.path.join(os.getcwd(), filename)
    else:
        output = os.path.expanduser(output)
    shutil.move(archive.file, output)
    return output