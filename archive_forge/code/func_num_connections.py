import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
@property
def num_connections(self) -> int:
    """Return the number of active connections allowed at a time.

        We occasionally see issues where too many concurrent connections may lead to
        failed SSH connections when there are too many TPU hosts.

        We utilize this property to cap the maximum number of active connections
        at a time until a proper fix is found.

        """
    num_max_concurrent_active_connections = ray_constants.env_integer(ray_constants.RAY_TPU_MAX_CONCURRENT_CONNECTIONS_ENV_VAR, default=16)
    return min(self._num_workers, num_max_concurrent_active_connections)