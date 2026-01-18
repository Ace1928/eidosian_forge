import copy
import logging
import operator
import threading
import time
import traceback
from typing import Any, Dict, Optional
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import hash_launch_conf
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.tags import (
def launch_node(self, config: Dict[str, Any], count: int, node_type: str) -> Optional[Dict]:
    self.log('Got {} nodes to launch.'.format(count))
    created_nodes = self._launch_node(config, count, node_type)
    self.pending.dec(node_type, count)
    return created_nodes