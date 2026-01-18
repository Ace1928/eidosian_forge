import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def kind_and_type(pod: Dict[str, Any]) -> Tuple[NodeKind, NodeType]:
    """Determine Ray node kind (head or workers) and node type (worker group name)
    from a Ray pod's labels.
    """
    labels = pod['metadata']['labels']
    if labels[KUBERAY_LABEL_KEY_KIND] == KUBERAY_KIND_HEAD:
        kind = NODE_KIND_HEAD
        type = KUBERAY_TYPE_HEAD
    else:
        kind = NODE_KIND_WORKER
        type = labels[KUBERAY_LABEL_KEY_TYPE]
    return (kind, type)