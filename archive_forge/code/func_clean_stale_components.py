import json
import logging
import os
import re
import threading
import time
import traceback
from collections import namedtuple
from typing import List, Tuple, Any, Dict
from prometheus_client.core import (
from opencensus.metrics.export.value import ValueDouble
from opencensus.stats import aggregation
from opencensus.stats import measure as measure_module
from opencensus.stats.view_manager import ViewManager
from opencensus.stats.stats_recorder import StatsRecorder
from opencensus.stats.base_exporter import StatsExporter
from prometheus_client.core import Metric as PrometheusMetric
from opencensus.stats.aggregation_data import (
from opencensus.stats.view import View
from opencensus.tags import tag_key as tag_key_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.tags import tag_value as tag_value_module
import ray
from ray._raylet import GcsClient
from ray.core.generated.metrics_pb2 import Metric
def clean_stale_components(self):
    """Clean up stale components.

        Stale means the component is dead or unresponsive.

        Stale components won't be reported to Prometheus anymore.
        """
    with self._components_lock:
        stale_components = []
        stale_component_ids = []
        for id, component in self._components.items():
            elapsed = time.monotonic() - component.last_reported_time
            if elapsed > self._component_timeout_s:
                stale_component_ids.append(id)
                logger.info('Metrics from a worker ({}) is cleaned up due to timeout. Time since last report {}s'.format(id, elapsed))
        for id in stale_component_ids:
            stale_components.append(self._components.pop(id))
        return stale_components