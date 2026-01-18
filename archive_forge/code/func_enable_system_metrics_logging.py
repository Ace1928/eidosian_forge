from mlflow.environment_variables import (
from mlflow.utils.annotations import experimental
@experimental
def enable_system_metrics_logging():
    """Enable system metrics logging globally.

    Calling this function will enable system metrics logging globally, but users can still opt out
    system metrics logging for individual runs by `mlflow.start_run(log_system_metrics=False)`.
    """
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(True)