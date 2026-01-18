from mlflow.environment_variables import (
from mlflow.utils.annotations import experimental
@experimental
def set_system_metrics_sampling_interval(interval):
    """Set the system metrics sampling interval.

    Every `interval` seconds, the system metrics will be collected. By default `interval=10`.
    """
    if interval is None:
        MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL.unset()
    else:
        MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL.set(interval)