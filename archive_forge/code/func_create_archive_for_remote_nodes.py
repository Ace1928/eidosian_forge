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
def create_archive_for_remote_nodes(archive: Archive, remote_nodes: Sequence[Node], parameters: GetParameters):
    """Create an archive combining data from the remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        archive: Archive object to add remote data to.
        remote_nodes (Sequence[Node]): Sequence of remote nodes.
        parameters: Parameters (settings) for getting data.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SSH_WORKERS) as executor:
        for remote_node in remote_nodes:
            executor.submit(create_and_add_remote_data_to_local_archive, archive=archive, remote_node=remote_node, parameters=parameters)
    return archive