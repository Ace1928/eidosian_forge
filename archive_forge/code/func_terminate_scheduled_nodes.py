import copy
import logging
import math
import operator
import os
import queue
import subprocess
import threading
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import yaml
import ray
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.legacy_info_string import legacy_log_info_string
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.local.node_provider import (
from ray.autoscaler._private.node_launcher import BaseNodeLauncher, NodeLauncher
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.node_tracker import NodeTracker
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler._private.resource_demand_scheduler import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.exceptions import RpcError
def terminate_scheduled_nodes(self):
    """Terminate scheduled nodes and clean associated autoscaler state."""
    assert self.provider
    assert self.non_terminated_nodes
    if not self.nodes_to_terminate:
        return
    if self.worker_rpc_drain:
        self.drain_nodes_via_gcs(self.nodes_to_terminate)
    self.provider.terminate_nodes(self.nodes_to_terminate)
    for node in self.nodes_to_terminate:
        self.node_tracker.untrack(node)
        self.prom_metrics.stopped_nodes.inc()
    self.non_terminated_nodes.remove_terminating_nodes(self.nodes_to_terminate)
    self.nodes_to_terminate = []