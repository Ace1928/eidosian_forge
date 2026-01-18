import time
from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib
@tf_contextlib.contextmanager
def monitored_timer(metric_name, state_tracker=None):
    """Monitor the execution time and collect it into the specified metric."""
    if not enable_metrics:
        yield
    else:
        if not _METRICS_MAPPING:
            _init()
        start_time = time.time()
        start_state = state_tracker() if state_tracker else None
        yield
        duration_sec = time.time() - start_time
        if state_tracker is None or state_tracker() != start_state:
            metric = _METRICS_MAPPING[metric_name]
            metric.get_cell().add(duration_sec)