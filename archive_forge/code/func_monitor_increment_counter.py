import time
from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib
def monitor_increment_counter(metric_name):
    if not enable_metrics:
        return
    else:
        if not _METRICS_MAPPING:
            _init()
        metric = _METRICS_MAPPING[metric_name]
        metric.get_cell().increase_by(1)