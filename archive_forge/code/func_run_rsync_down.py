import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
def run_rsync_down(self, *args, **kwargs) -> None:
    """Rsync files down from the cluster node.

        Args:
            source: The (remote) source directory or file.
            target: The (local) destination path.
        """
    with ThreadPoolExecutor(self.num_connections) as executor:
        executor.map(lambda i: self._command_runners[i].run_rsync_down(*args, **kwargs), range(self._num_workers))