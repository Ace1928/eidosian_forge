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
@classmethod
def from_node_config(cls, node_config: NodeTypeConfig, status: SchedulingNodeStatus=SchedulingNodeStatus.TO_LAUNCH) -> 'SchedulingNode':
    """
        Create a scheduling node from a node config.
        """
    return cls(node_type=node_config.name, total_resources=dict(node_config.resources), available_resources=dict(node_config.resources), labels=dict(node_config.labels), status=status)