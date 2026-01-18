import concurrent.futures
from threading import RLock
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
def flush_metrics_queue():
    """Flush the metric queue and log contents in batches to MLflow.

    Queue is divided into batches according to run id.
    """
    try:
        acquired_lock = _metrics_queue_lock.acquire(blocking=False)
        if acquired_lock:
            client = MlflowClient()
            snapshot = _metrics_queue[:]
            for item in snapshot:
                _metrics_queue.remove(item)
            metrics_by_run = _assoc_list_to_map(snapshot)
            for run_id, metrics in metrics_by_run.items():
                client.log_batch(run_id, metrics=metrics, params=[], tags=[])
    finally:
        if acquired_lock:
            _metrics_queue_lock.release()