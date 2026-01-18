from typing import Optional
from ray._raylet import GcsClient
from ray.serve._private.cluster_node_info_cache import (
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.utils import get_head_node_id
def create_cluster_node_info_cache(gcs_client: GcsClient) -> ClusterNodeInfoCache:
    return DefaultClusterNodeInfoCache(gcs_client)