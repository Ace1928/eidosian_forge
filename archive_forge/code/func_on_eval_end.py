import paddle
import mlflow
from mlflow.utils.autologging_utils import (
def on_eval_end(self, logs=None):
    eval_logs = {'eval_' + key: metric[0] if isinstance(metric, list) else metric for key, metric in logs.items()}
    self._log_metrics(eval_logs, self.epoch)