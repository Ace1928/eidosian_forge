import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray  # noqa: F401
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
def get_all_local_data(archive: Archive, parameters: GetParameters):
    """Get all local data.

    Gets:
        - The Ray logs of the latest session
        - The currently installed pip packages

    Args:
        archive: Archive object to add meta files to.
        parameters: Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    if not archive.is_open:
        archive.open()
    if parameters.logs:
        try:
            get_local_ray_logs(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.debug_state:
        try:
            get_local_debug_state(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.pip:
        try:
            get_local_pip_packages(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.processes:
        try:
            get_local_ray_processes(archive=archive, processes=parameters.processes_list, verbose=parameters.processes_verbose)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    return archive