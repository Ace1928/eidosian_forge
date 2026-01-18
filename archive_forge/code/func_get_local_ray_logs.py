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
def get_local_ray_logs(archive: Archive, exclude: Optional[Sequence[str]]=None, session_log_dir: str='/tmp/ray/session_latest') -> Archive:
    """Copy local log files into an archive.

    Args:
        archive: Archive object to add log files to.
        exclude (Sequence[str]): Sequence of regex patterns. Files that match
            any of these patterns will not be included in the archive.
        session_dir: Path to the Ray session files. Defaults to
            ``/tmp/ray/session_latest``

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()
    exclude = exclude or []
    session_log_dir = os.path.join(os.path.expanduser(session_log_dir), 'logs')
    with archive.subdir('logs', root=session_log_dir) as sd:
        for root, dirs, files in os.walk(session_log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start=session_log_dir)
                if any((re.match(pattern, rel_path) for pattern in exclude)):
                    continue
                sd.add(file_path)
    return archive