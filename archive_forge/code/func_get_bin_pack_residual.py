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
def get_bin_pack_residual(node_resources: List[ResourceDict], resource_demands: List[ResourceDict], strict_spread: bool=False) -> (List[ResourceDict], List[ResourceDict]):
    """Return a subset of resource_demands that cannot fit in the cluster.

    TODO(ekl): this currently does not guarantee the resources will be packed
    correctly by the Ray scheduler. This is only possible once the Ray backend
    supports a placement groups API.

    Args:
        node_resources (List[ResourceDict]): List of resources per node.
        resource_demands (List[ResourceDict]): List of resource bundles that
            need to be bin packed onto the nodes.
        strict_spread: If true, each element in resource_demands must be
            placed on a different entry in `node_resources`.

    Returns:
        List[ResourceDict]: the residual list resources that do not fit.
        List[ResourceDict]: The updated node_resources after the method.
    """
    unfulfilled = []
    nodes = copy.deepcopy(node_resources)
    used = []
    for demand in sorted(resource_demands, key=lambda demand: (len(demand.values()), sum(demand.values()), sorted(demand.items())), reverse=True):
        found = False
        node = None
        for i in range(len(nodes)):
            node = nodes[i]
            if _fits(node, demand):
                found = True
                if strict_spread:
                    used.append(node)
                    del nodes[i]
                break
        if found and node:
            _inplace_subtract(node, demand)
        else:
            unfulfilled.append(demand)
    return (unfulfilled, nodes + used)