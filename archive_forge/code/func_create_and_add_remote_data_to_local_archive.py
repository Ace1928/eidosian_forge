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
def create_and_add_remote_data_to_local_archive(archive: Archive, remote_node: Node, parameters: GetParameters):
    """Create and get data from remote node and add to local archive.

    Args:
        archive: Archive object to add remote data to.
        remote_node: Remote node to gather archive from.
        parameters: Parameters (settings) for getting data.

    Returns:
        Open archive object.
    """
    tmp = create_and_get_archive_from_remote_node(remote_node, parameters)
    if not archive.is_open:
        archive.open()
    cat = 'node' if not remote_node.is_head else 'head'
    with archive.subdir('', root=os.path.dirname(tmp)) as sd:
        sd.add(tmp, arcname=f'ray_{cat}_{remote_node.host}.tar.gz')
    return archive