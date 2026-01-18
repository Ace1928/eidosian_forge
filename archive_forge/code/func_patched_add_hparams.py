import time
from contextlib import contextmanager
import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.metrics_queue import (
def patched_add_hparams(original, self, hparam_dict, metric_dict, *args, **kwargs):
    """use a synchronous call here since this is going to get called very infrequently."""
    run = mlflow.active_run()
    if not DISABLED and run is not None and hparam_dict:
        run_id = run.info.run_id
        params_arr = [Param(key, str(value)) for key, value in hparam_dict.items()]
        metrics_arr = [Metric(key, value, int(time.time() * 1000), 0) for key, value in metric_dict.items()]
        MlflowClient().log_batch(run_id=run_id, metrics=metrics_arr, params=params_arr, tags=[])
    return original(self, hparam_dict, metric_dict, *args, **kwargs)