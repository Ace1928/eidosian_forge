import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib
import urllib.parse
import warnings
import shutil
from datetime import datetime
from typing import Optional, Set, List, Tuple
import click
import psutil
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._private.utils import (
from ray._private.internal_api import memory_summary
from ray._private.storage import _load_class
from ray._private.usage import usage_lib
from ray.autoscaler._private.cli_logger import add_click_logging_options, cf, cli_logger
from ray.autoscaler._private.commands import (
from ray.autoscaler._private.constants import RAY_PROCESSES
from ray.autoscaler._private.fake_multi_node.node_provider import FAKE_HEAD_NODE_ID
from ray.util.annotations import PublicAPI
@cli.command()
@click.argument('cluster_config_file', required=False, type=str)
@click.option('--host', '-h', required=False, type=str, help='Single or list of hosts, separated by comma.')
@click.option('--ssh-user', '-U', required=False, type=str, default=None, help='Username of the SSH user.')
@click.option('--ssh-key', '-K', required=False, type=str, default=None, help='Path to the SSH key file.')
@click.option('--docker', '-d', required=False, type=str, default=None, help='Name of the docker container, if applicable.')
@click.option('--local', '-L', required=False, type=bool, is_flag=True, default=None, help='Also include information about the local node.')
@click.option('--output', '-o', required=False, type=str, default=None, help='Output file.')
@click.option('--logs/--no-logs', is_flag=True, default=True, help='Collect logs from ray session dir')
@click.option('--debug-state/--no-debug-state', is_flag=True, default=True, help='Collect debug_state.txt from ray log dir')
@click.option('--pip/--no-pip', is_flag=True, default=True, help='Collect installed pip packages')
@click.option('--processes/--no-processes', is_flag=True, default=True, help='Collect info on running processes')
@click.option('--processes-verbose/--no-processes-verbose', is_flag=True, default=True, help='Increase process information verbosity')
@click.option('--tempfile', '-T', required=False, type=str, default=None, help='Temporary file to use')
def cluster_dump(cluster_config_file: Optional[str]=None, host: Optional[str]=None, ssh_user: Optional[str]=None, ssh_key: Optional[str]=None, docker: Optional[str]=None, local: Optional[bool]=None, output: Optional[str]=None, logs: bool=True, debug_state: bool=True, pip: bool=True, processes: bool=True, processes_verbose: bool=False, tempfile: Optional[str]=None):
    """Get log data from one or more nodes.

    Best used with Ray cluster configs:

        ray cluster-dump [cluster.yaml]

    Include the --local flag to also collect and include data from the
    local node.

    Missing fields will be tried to be auto-filled.

    You can also manually specify a list of hosts using the
    ``--host <host1,host2,...>`` parameter.
    """
    archive_path = get_cluster_dump_archive(cluster_config_file=cluster_config_file, host=host, ssh_user=ssh_user, ssh_key=ssh_key, docker=docker, local=local, output=output, logs=logs, debug_state=debug_state, pip=pip, processes=processes, processes_verbose=processes_verbose, tempfile=tempfile)
    if archive_path:
        click.echo(f'Created archive: {archive_path}')
    else:
        click.echo('Could not create archive.')