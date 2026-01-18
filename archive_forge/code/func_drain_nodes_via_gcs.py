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
def drain_nodes_via_gcs(self, provider_node_ids_to_drain: List[NodeID]):
    """Send an RPC request to the GCS to drain (prepare for termination)
        the nodes with the given node provider ids.

        note: The current implementation of DrainNode on the GCS side is to
        de-register and gracefully shut down the Raylets. In the future,
        the behavior may change to better reflect the name "Drain."
        See https://github.com/ray-project/ray/pull/19350.
        """
    assert self.provider
    node_ips = set()
    failed_ip_fetch = False
    for provider_node_id in provider_node_ids_to_drain:
        try:
            ip = self.provider.internal_ip(provider_node_id)
            node_ips.add(ip)
        except Exception:
            logger.exception(f'Failed to get ip of node with id {provider_node_id} during scale-down.')
            failed_ip_fetch = True
    if failed_ip_fetch:
        self.prom_metrics.drain_node_exceptions.inc()
    connected_node_ips = node_ips & self.load_metrics.raylet_id_by_ip.keys()
    raylet_ids_to_drain = {self.load_metrics.raylet_id_by_ip[ip] for ip in connected_node_ips}
    if not raylet_ids_to_drain:
        return
    logger.info(f'Draining {len(raylet_ids_to_drain)} raylet(s).')
    try:
        drained_raylet_ids = set(self.gcs_client.drain_nodes(raylet_ids_to_drain, timeout=5))
        failed_to_drain = raylet_ids_to_drain - drained_raylet_ids
        if failed_to_drain:
            self.prom_metrics.drain_node_exceptions.inc()
            logger.error(f'Failed to drain {len(failed_to_drain)} raylet(s).')
    except RpcError as e:
        if e.rpc_code == ray._raylet.GRPC_STATUS_CODE_UNIMPLEMENTED:
            pass
        else:
            self.prom_metrics.drain_node_exceptions.inc()
            logger.exception('Failed to drain Ray nodes. Traceback follows.')
    except Exception:
        self.prom_metrics.drain_node_exceptions.inc()
        logger.exception('Failed to drain Ray nodes. Traceback follows.')