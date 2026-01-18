import concurrent.futures
from threading import RLock
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
def add_to_metrics_queue(key, value, step, time, run_id):
    """Add a metric to the metric queue.

    Flush the queue if it exceeds max size.

    Args:
        key: string, the metrics key,
        value: float, the metrics value.
        step: int, the step of current metric.
        time: int, the timestamp of current metric.
        run_id: string, the run id of the associated mlflow run.
    """
    met = Metric(key=key, value=value, timestamp=time, step=step)
    _metrics_queue.append((run_id, met))
    if len(_metrics_queue) > _MAX_METRIC_QUEUE_SIZE:
        _thread_pool.submit(flush_metrics_queue)