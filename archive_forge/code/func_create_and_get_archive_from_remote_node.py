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
def create_and_get_archive_from_remote_node(remote_node: Node, parameters: GetParameters, script_path: str='ray') -> Optional[str]:
    """Create an archive containing logs on a remote node and transfer.

    This will call ``ray local-dump --stream`` on the remote
    node. The resulting file will be saved locally in a temporary file and
    returned.

    Args:
        remote_node: Remote node to gather archive from.
        script_path: Path to this script on the remote node.
        parameters: Parameters (settings) for getting data.

    Returns:
        Path to a temporary file containing the node's collected data.

    """
    cmd = ['ssh', '-o StrictHostKeyChecking=no', '-o UserKnownHostsFile=/dev/null', '-o LogLevel=ERROR', '-i', remote_node.ssh_key, f'{remote_node.ssh_user}@{remote_node.host}']
    if remote_node.docker_container:
        cmd += ['docker', 'exec', remote_node.docker_container]
    collect_cmd = [script_path, 'local-dump', '--stream']
    collect_cmd += ['--logs'] if parameters.logs else ['--no-logs']
    collect_cmd += ['--debug-state'] if parameters.debug_state else ['--no-debug-state']
    collect_cmd += ['--pip'] if parameters.pip else ['--no-pip']
    collect_cmd += ['--processes'] if parameters.processes else ['--no-processes']
    if parameters.processes:
        collect_cmd += ['--processes-verbose'] if parameters.processes_verbose else ['--no-proccesses-verbose']
    cmd += ['/bin/bash', '-c', _wrap(collect_cmd, quotes='"')]
    cat = 'node' if not remote_node.is_head else 'head'
    cli_logger.print(f'Collecting data from remote node: {remote_node.host}')
    tmp = tempfile.mkstemp(prefix=f'ray_{cat}_{remote_node.host}_', suffix='.tar.gz')[1]
    with open(tmp, 'wb') as fp:
        try:
            subprocess.check_call(cmd, stdout=fp, stderr=sys.stderr)
        except subprocess.CalledProcessError as exc:
            raise RemoteCommandFailed(f'Gathering logs from remote node failed: {' '.join(cmd)}') from exc
    return tmp