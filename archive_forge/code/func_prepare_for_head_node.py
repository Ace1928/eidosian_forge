import logging
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.util.annotations import DeveloperAPI
def prepare_for_head_node(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a new cluster config with custom configs for head node."""
    return cluster_config