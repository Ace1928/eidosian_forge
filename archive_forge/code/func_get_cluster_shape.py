import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
def get_cluster_shape(self) -> Dict[NodeType, int]:
    cluster_shape = defaultdict(int)
    for node in self._nodes:
        cluster_shape[node.node_type] += 1
    return cluster_shape