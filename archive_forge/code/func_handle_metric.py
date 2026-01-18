import json
import logging
import math
import numbers
import time
from collections import defaultdict
from queue import Queue
from threading import Event
from typing import (
from wandb.proto.wandb_internal_pb2 import (
from ..interface.interface_queue import InterfaceQueue
from ..lib import handler_util, proto_util, tracelog, wburls
from . import context, sample, tb_watcher
from .settings_static import SettingsStatic
from .system.system_monitor import SystemMonitor
def handle_metric(self, record: Record) -> None:
    """Handle MetricRecord.

        Walkthrough of the life of a MetricRecord:

        Metric defined:
        - run.define_metric() parses arguments create wandb_metric.Metric
        - build MetricRecord publish to interface
        - handler (this function) keeps list of metrics published:
          - self._metric_defines: Fully defined metrics
          - self._metric_globs: metrics that have a wildcard
        - dispatch writer and sender thread
          - writer: records are saved to persistent store
          - sender: fully defined metrics get mapped into metadata for UI

        History logged:
        - handle_history
        - check if metric matches _metric_defines
        - if not, check if metric matches _metric_globs
        - if _metric globs match, generate defined metric and call _handle_metric

        Args:
            record (Record): Metric record to process
        """
    if record.metric.name:
        self._handle_defined_metric(record)
    elif record.metric.glob_name:
        self._handle_glob_metric(record)