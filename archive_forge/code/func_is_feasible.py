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
def is_feasible(self, bundle: ResourceDict) -> bool:
    for node_type, config in self.node_types.items():
        max_of_type = config.get('max_workers', 0)
        node_resources = config['resources']
        if (node_type == self.head_node_type or max_of_type > 0) and _fits(node_resources, bundle):
            return True
    return False