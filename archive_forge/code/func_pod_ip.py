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
def pod_ip(pod: Dict[str, Any]) -> NodeIP:
    return pod['status'].get('podIP', 'IP not yet assigned')