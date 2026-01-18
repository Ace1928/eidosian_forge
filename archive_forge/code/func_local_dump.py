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
@cli.command(hidden=True)
@click.option('--stream', '-S', required=False, type=bool, is_flag=True, default=False, help='If True, will stream the binary archive contents to stdout')
@click.option('--output', '-o', required=False, type=str, default=None, help='Output file.')
@click.option('--logs/--no-logs', is_flag=True, default=True, help='Collect logs from ray session dir')
@click.option('--debug-state/--no-debug-state', is_flag=True, default=True, help='Collect debug_state.txt from ray session dir')
@click.option('--pip/--no-pip', is_flag=True, default=True, help='Collect installed pip packages')
@click.option('--processes/--no-processes', is_flag=True, default=True, help='Collect info on running processes')
@click.option('--processes-verbose/--no-processes-verbose', is_flag=True, default=True, help='Increase process information verbosity')
@click.option('--tempfile', '-T', required=False, type=str, default=None, help='Temporary file to use')
def local_dump(stream: bool=False, output: Optional[str]=None, logs: bool=True, debug_state: bool=True, pip: bool=True, processes: bool=True, processes_verbose: bool=False, tempfile: Optional[str]=None):
    """Collect local data and package into an archive.

    Usage:

        ray local-dump [--stream/--output file]

    This script is called on remote nodes to fetch their data.
    """
    get_local_dump_archive(stream=stream, output=output, logs=logs, debug_state=debug_state, pip=pip, processes=processes, processes_verbose=processes_verbose, tempfile=tempfile)