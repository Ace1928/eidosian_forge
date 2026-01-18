import logging
import threading
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor
from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor
def publish_metrics(self, metrics):
    """Log collected metrics to MLflow."""
    prefix = self._metrics_prefix + (self.node_id + '/' if self.node_id else '')
    metrics = {prefix + k: v for k, v in metrics.items()}
    self.mlflow_logger.record_metrics(metrics, self._logging_step)
    self._logging_step += 1
    for monitor in self.monitors:
        monitor.clear_metrics()