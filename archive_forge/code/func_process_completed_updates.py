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
def process_completed_updates(self):
    """Clean up completed NodeUpdaterThreads."""
    completed_nodes = []
    for node_id, updater in self.updaters.items():
        if not updater.is_alive():
            completed_nodes.append(node_id)
    if completed_nodes:
        failed_nodes = []
        for node_id in completed_nodes:
            updater = self.updaters[node_id]
            if updater.exitcode == 0:
                self.num_successful_updates[node_id] += 1
                self.prom_metrics.successful_updates.inc()
                if updater.for_recovery:
                    self.prom_metrics.successful_recoveries.inc()
                if updater.update_time:
                    self.prom_metrics.worker_update_time.observe(updater.update_time)
                self.load_metrics.mark_active(self.provider.internal_ip(node_id))
            else:
                failed_nodes.append(node_id)
                self.num_failed_updates[node_id] += 1
                self.prom_metrics.failed_updates.inc()
                if updater.for_recovery:
                    self.prom_metrics.failed_recoveries.inc()
                self.node_tracker.untrack(node_id)
            del self.updaters[node_id]
        if failed_nodes:
            for node_id in failed_nodes:
                if node_id in self.non_terminated_nodes.worker_ids:
                    self.schedule_node_termination(node_id, 'launch failed', logger.error)
                else:
                    logger.warning(f'StandardAutoscaler: {node_id}: Failed to update node. Node has already been terminated.')
            self.terminate_scheduled_nodes()