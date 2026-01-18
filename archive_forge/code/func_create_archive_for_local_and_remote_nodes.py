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
def create_archive_for_local_and_remote_nodes(archive: Archive, remote_nodes: Sequence[Node], parameters: GetParameters):
    """Create an archive combining data from the local and remote nodes.

    This will parallelize calls to get data from remote nodes.

    Args:
        archive: Archive object to add data to.
        remote_nodes (Sequence[Node]): Sequence of remote nodes.
        parameters: Parameters (settings) for getting data.

    Returns:
        Open archive object.

    """
    if not archive.is_open:
        archive.open()
    try:
        create_and_add_local_data_to_local_archive(archive, parameters)
    except CommandFailed as exc:
        cli_logger.error(exc)
    create_archive_for_remote_nodes(archive, remote_nodes, parameters)
    cli_logger.print(f'Collected data from local node and {len(remote_nodes)} remote nodes.')
    return archive