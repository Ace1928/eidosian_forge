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
def kill_procs(force: bool, grace_period: int, processes_to_kill: List[str]) -> Tuple[int, int, List[psutil.Process]]:
    """Find all processes from `processes_to_kill` and terminate them.

        Unless `force` is specified, it gracefully kills processes. If
        processes are not cleaned within `grace_period`, it force kill all
        remaining processes.

        Returns:
            total_procs_found: Total number of processes found from
                `processes_to_kill` is added.
            total_procs_stopped: Total number of processes gracefully
                stopped from `processes_to_kill` is added.
            procs_not_gracefully_killed: If processes are not killed
                gracefully, they are added here.
        """
    process_infos = []
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            process_infos.append((proc, proc.name(), proc.cmdline()))
        except psutil.Error:
            pass
    stopped = []
    for keyword, filter_by_cmd in processes_to_kill:
        if filter_by_cmd and is_linux and (len(keyword) > 15):
            msg = 'The filter string should not be more than {} characters. Actual length: {}. Filter: {}'.format(15, len(keyword), keyword)
            raise ValueError(msg)
        found = []
        for candidate in process_infos:
            proc, proc_cmd, proc_args = candidate
            corpus = proc_cmd if filter_by_cmd else subprocess.list2cmdline(proc_args)
            if keyword in corpus:
                found.append(candidate)
        for proc, proc_cmd, proc_args in found:
            proc_string = str(subprocess.list2cmdline(proc_args))
            try:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                if force:
                    cli_logger.verbose('Killed `{}` {} ', cf.bold(proc_string), cf.dimmed('(via SIGKILL)'))
                else:
                    cli_logger.verbose('Send termination request to `{}` {}', cf.bold(proc_string), cf.dimmed('(via SIGTERM)'))
                stopped.append(proc)
            except psutil.NoSuchProcess:
                cli_logger.verbose('Attempted to stop `{}`, but process was already dead.', cf.bold(proc_string))
            except (psutil.Error, OSError) as ex:
                cli_logger.error('Could not terminate `{}` due to {}', cf.bold(proc_string), str(ex))
    stopped, alive = psutil.wait_procs(stopped, timeout=0)
    procs_to_kill = stopped + alive
    total_found = len(procs_to_kill)
    gone_procs = set()

    def on_terminate(proc):
        gone_procs.add(proc)
        cli_logger.print(f'{len(gone_procs)}/{total_found} stopped.', end='\r')
    stopped, alive = psutil.wait_procs(procs_to_kill, timeout=grace_period, callback=on_terminate)
    total_stopped = len(stopped)
    for proc in alive:
        proc.kill()
    psutil.wait_procs(alive, timeout=2)
    return (total_found, total_stopped, alive)