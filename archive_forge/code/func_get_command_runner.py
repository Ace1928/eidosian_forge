import copy
import logging
import time
from functools import wraps
from threading import RLock
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import googleapiclient
from ray.autoscaler._private.gcp.config import (
from ray.autoscaler._private.gcp.node import GCPTPU  # noqa
from ray.autoscaler._private.gcp.node import (
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
def get_command_runner(self, log_prefix: str, node_id: str, auth_config: Dict[str, Any], cluster_name: str, process_runner: ModuleType, use_internal_ip: bool, docker_config: Optional[Dict[str, Any]]=None) -> CommandRunnerInterface:
    """Returns a TPU command runner as applicable."""
    resource = self._get_resource_depending_on_node_name(node_id)
    instance = resource.get_instance(node_id)
    common_args = {'docker_config': docker_config, 'log_prefix': log_prefix, 'node_id': node_id, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': use_internal_ip}
    if GCPNodeType.TPU in self.resources and resource == self.resources[GCPNodeType.TPU]:
        return TPUCommandRunner(instance=instance, provider=self, **common_args)
    else:
        return super().get_command_runner(**common_args)