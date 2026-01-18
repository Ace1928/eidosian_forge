import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from ray.autoscaler._private import commands
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent  # noqa: F401
from ray.autoscaler._private.event_system import global_event_system  # noqa: F401
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def get_docker_host_mount_location(cluster_name: str) -> str:
    """Return host path that Docker mounts attach to."""
    docker_mount_prefix = '/tmp/ray_tmp_mount/{cluster_name}'
    return docker_mount_prefix.format(cluster_name=cluster_name)