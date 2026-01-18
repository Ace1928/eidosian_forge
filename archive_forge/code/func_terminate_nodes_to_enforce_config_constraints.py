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
def terminate_nodes_to_enforce_config_constraints(self, now: float):
    """Terminates nodes to enforce constraints defined by the autoscaling
        config.

        (1) Terminates nodes in excess of `max_workers`.
        (2) Terminates nodes idle for longer than `idle_timeout_minutes`.
        (3) Terminates outdated nodes,
                namely nodes whose configs don't match `node_config` for the
                relevant node type.

        Avoids terminating non-outdated nodes required by
        autoscaler.sdk.request_resources().
        """
    assert self.non_terminated_nodes
    assert self.provider
    last_used = self.load_metrics.last_used_time_by_ip
    horizon = now - 60 * self.config['idle_timeout_minutes']
    sorted_node_ids = self._sort_based_on_last_used(self.non_terminated_nodes.worker_ids, last_used)
    nodes_not_allowed_to_terminate: FrozenSet[NodeID] = {}
    if self.load_metrics.get_resource_requests():
        nodes_not_allowed_to_terminate = self._get_nodes_needed_for_request_resources(sorted_node_ids)
    node_type_counts = defaultdict(int)

    def keep_node(node_id: NodeID) -> None:
        assert self.provider
        tags = self.provider.node_tags(node_id)
        if TAG_RAY_USER_NODE_TYPE in tags:
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            node_type_counts[node_type] += 1
    nodes_we_could_terminate: List[NodeID] = []
    for node_id in sorted_node_ids:
        should_keep_or_terminate, reason = self._keep_worker_of_node_type(node_id, node_type_counts)
        if should_keep_or_terminate == KeepOrTerminate.terminate:
            self.schedule_node_termination(node_id, reason, logger.info)
            continue
        if (should_keep_or_terminate == KeepOrTerminate.keep or node_id in nodes_not_allowed_to_terminate) and self.launch_config_ok(node_id):
            keep_node(node_id)
            continue
        node_ip = self.provider.internal_ip(node_id)
        if node_ip in last_used and last_used[node_ip] < horizon:
            self.schedule_node_termination(node_id, 'idle', logger.info)
            formatted_last_used_time = time.asctime(time.localtime(last_used[node_ip]))
            logger.info(f'Node last used: {formatted_last_used_time}.')
        elif not self.launch_config_ok(node_id):
            self.schedule_node_termination(node_id, 'outdated', logger.info)
        else:
            keep_node(node_id)
            nodes_we_could_terminate.append(node_id)
    num_workers = len(self.non_terminated_nodes.worker_ids)
    num_extra_nodes_to_terminate = num_workers - len(self.nodes_to_terminate) - self.config['max_workers']
    if num_extra_nodes_to_terminate > len(nodes_we_could_terminate):
        logger.warning(f'StandardAutoscaler: trying to terminate {num_extra_nodes_to_terminate} nodes, while only {len(nodes_we_could_terminate)} are safe to terminate. Inconsistent config is likely.')
        num_extra_nodes_to_terminate = len(nodes_we_could_terminate)
    if num_extra_nodes_to_terminate > 0:
        extra_nodes_to_terminate = nodes_we_could_terminate[-num_extra_nodes_to_terminate:]
        for node_id in extra_nodes_to_terminate:
            self.schedule_node_termination(node_id, 'max workers', logger.info)
    self.terminate_scheduled_nodes()