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
def get_local_ray_processes(archive: Archive, processes: Optional[List[Tuple[str, bool]]]=None, verbose: bool=False):
    """Get the status of all the relevant ray processes.
    Args:
        archive: Archive object to add process info files to.
        processes: List of processes to get information on. The first
            element of the tuple is a string to filter by, and the second
            element is a boolean indicating if we should filter by command
            name (True) or command line including parameters (False)
        verbose: If True, show entire executable command line.
            If False, show just the first term.
    Returns:
        Open archive object.
    """
    if not processes:
        from ray.autoscaler._private.constants import RAY_PROCESSES
        processes = RAY_PROCESSES
    process_infos = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            with process.oneshot():
                cmdline = ' '.join(process.cmdline())
                process_infos.append(({'executable': cmdline if verbose else cmdline.split('--', 1)[0][:-1], 'name': process.name(), 'pid': process.pid, 'status': process.status()}, process.cmdline()))
        except Exception as exc:
            raise LocalCommandFailed(exc) from exc
    relevant_processes = {}
    for process_dict, cmdline in process_infos:
        for keyword, filter_by_cmd in processes:
            if filter_by_cmd:
                corpus = process_dict['name']
            else:
                corpus = subprocess.list2cmdline(cmdline)
            if keyword in corpus and process_dict['pid'] not in relevant_processes:
                relevant_processes[process_dict['pid']] = process_dict
    with tempfile.NamedTemporaryFile('wt') as fp:
        for line in relevant_processes.values():
            fp.writelines([yaml.dump(line), '\n'])
        fp.flush()
        with archive.subdir('meta') as sd:
            sd.add(fp.name, 'process_info.txt')
    return archive