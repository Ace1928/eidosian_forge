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
def create_and_add_local_data_to_local_archive(archive: Archive, parameters: GetParameters):
    """Create and get data from this node and add to archive.

    Args:
        archive: Archive object to add remote data to.
        parameters: Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    with Archive() as local_data_archive:
        get_all_local_data(local_data_archive, parameters)
    if not archive.is_open:
        archive.open()
    with archive.subdir('', root=os.path.dirname(local_data_archive.file)) as sd:
        sd.add(local_data_archive.file, arcname='local_node.tar.gz')
    os.remove(local_data_archive.file)
    return archive