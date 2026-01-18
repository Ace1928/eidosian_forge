import time
from contextlib import contextmanager
import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.metrics_queue import (
def patched_add_event(original, self, event, *args, mlflow_log_every_n_step, **kwargs):
    run = mlflow.active_run()
    if not DISABLED and run is not None and (event.WhichOneof('what') == 'summary') and mlflow_log_every_n_step:
        summary = event.summary
        global_step = args[0] if len(args) > 0 else kwargs.get('global_step', None)
        global_step = global_step or 0
        for v in summary.value:
            if v.HasField('simple_value') and global_step % mlflow_log_every_n_step == 0:
                add_to_metrics_queue(key=v.tag, value=v.simple_value, step=global_step, time=int((event.wall_time or time.time()) * 1000), run_id=run.info.run_id)
    return original(self, event, *args, **kwargs)