import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def node_ids():
    """Get a list of the node ids in the cluster.

    For example, ["node:172.10.5.34", "node:172.42.3.77"]. These can be used
    as custom resources, e.g., {node_id: 1} to reserve the whole node, or
    {node_id: 0.001} to just force placement on the node.

    Returns:
        List of the node resource ids.
    """
    node_ids = []
    for node in nodes():
        for k, v in node['Resources'].items():
            if k.startswith(NODE_ID_PREFIX) and k != HEAD_NODE_RESOURCE_NAME:
                node_ids.append(k)
    return node_ids