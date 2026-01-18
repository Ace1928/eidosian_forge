import psutil
from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor
def aggregate_metrics(self):
    return dict(self._metrics)