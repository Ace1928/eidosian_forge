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
def get_local_debug_state(archive: Archive, session_dir: str='/tmp/ray/session_latest') -> Archive:
    """Copy local log files into an archive.

    Args:
        archive: Archive object to add log files to.
        session_dir: Path to the Ray session files. Defaults to
            ``/tmp/ray/session_latest``

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()
    session_dir = os.path.expanduser(session_dir)
    debug_state_file = os.path.join(session_dir, 'logs/debug_state.txt')
    if not os.path.exists(debug_state_file):
        raise LocalCommandFailed('No `debug_state.txt` file found.')
    with archive.subdir('', root=session_dir) as sd:
        sd.add(debug_state_file)
    return archive