from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Tuple
import ray
from ray._private.ray_constants import AUTOSCALER_NAMESPACE, AUTOSCALER_V2_ENABLED_KEY
from ray._private.utils import binary_to_hex
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import LoadMetricsSummary, format_info_string
from ray.autoscaler.v2.schema import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
from ray.experimental.internal_kv import _internal_kv_get, _internal_kv_initialized
@classmethod
def from_get_cluster_status_reply(cls, proto: GetClusterStatusReply, stats: Stats) -> ClusterStatus:
    active_nodes, idle_nodes, failed_nodes = cls._parse_nodes(proto.cluster_resource_state)
    pending_nodes = cls._parse_pending(proto.autoscaling_state)
    pending_launches, failed_launches = cls._parse_launch_requests(proto.autoscaling_state)
    cluster_resource_usage = cls._parse_cluster_resource_usage(proto.cluster_resource_state)
    resource_demands = cls._parse_resource_demands(proto.cluster_resource_state)
    stats = cls._parse_stats(proto, stats)
    return ClusterStatus(active_nodes=active_nodes, idle_nodes=idle_nodes, pending_launches=pending_launches, failed_launches=failed_launches, pending_nodes=pending_nodes, failed_nodes=failed_nodes, cluster_resource_usage=cluster_resource_usage, resource_demands=resource_demands, stats=stats)