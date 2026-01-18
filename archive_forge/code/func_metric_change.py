import logging
import threading
from .metrics_reporter import AbstractMetricsReporter
def metric_change(self, metric):
    with self._lock:
        category = self.get_category(metric)
        if category not in self._store:
            self._store[category] = {}
        self._store[category][metric.metric_name.name] = metric