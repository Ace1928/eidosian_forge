from typing import Optional
from ray._raylet import GcsClient
from ray.serve._private.cluster_node_info_cache import (
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.utils import get_head_node_id
def create_deployment_scheduler(cluster_node_info_cache: ClusterNodeInfoCache, head_node_id_override: Optional[str]=None) -> DeploymentScheduler:
    head_node_id = head_node_id_override or get_head_node_id()
    return DefaultDeploymentScheduler(cluster_node_info_cache, head_node_id)