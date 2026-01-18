import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set
from ray.autoscaler._private.node_launcher import BaseNodeLauncher
from ray.autoscaler.node_provider import NodeProvider as NodeProviderV1
from ray.autoscaler.tags import TAG_RAY_USER_NODE_TYPE
from ray.autoscaler.v2.instance_manager.config import NodeProviderConfig
from ray.core.generated.instance_manager_pb2 import Instance
def get_non_terminated_nodes(self):
    clould_instance_ids = self._provider.non_terminated_nodes({})
    return self.get_nodes_by_cloud_instance_id(clould_instance_ids)