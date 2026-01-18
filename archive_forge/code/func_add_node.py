import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
def add_node(node_type, available_resources=None):
    if node_type not in self.node_types:
        logger.error(f'''Missing entry for node_type {node_type} in cluster config: {self.node_types} under entry available_node_types. This node's resources will be ignored. If you are using an unmanaged node, manually set the {TAG_RAY_NODE_KIND} tag to "{NODE_KIND_UNMANAGED}" in your cloud provider's management console.''')
        return None
    available = copy.deepcopy(self.node_types[node_type]['resources'])
    if available_resources is not None:
        available = copy.deepcopy(available_resources)
    node_resources.append(available)
    node_type_counts[node_type] += 1