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
def proxy_export_metrics(self, metrics: List[Metric], worker_id_hex: str=None):
    """Proxy export metrics specified by a Opencensus Protobuf.

        This API is used to export metrics emitted from
        core components.

        Args:
            metrics: A list of protobuf Metric defined from OpenCensus.
            worker_id_hex: The worker ID it proxies metrics export. None
                if the metric is not from a worker (i.e., raylet, GCS).
        """
    with self._lock:
        if not self.view_manager:
            return
    self._proxy_export_metrics(metrics, worker_id_hex)