import logging
import threading
from .metrics_reporter import AbstractMetricsReporter
def metric_removal(self, metric):
    with self._lock:
        category = self.get_category(metric)
        metrics = self._store.get(category, {})
        removed = metrics.pop(metric.metric_name.name, None)
        if not metrics:
            self._store.pop(category, None)
        return removed