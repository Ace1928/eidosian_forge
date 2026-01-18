from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.deprecation import deprecated
@deprecated('2020-07-01', 'use `tf.profiler.experimental.client.trace`.')
def start_tracing(service_addr, logdir, duration_ms, worker_list='', include_dataset_ops=True, num_tracing_attempts=3):
    """Sends grpc requests to profiler server to perform on-demand profiling.

  This method will block caller thread until receives tracing result.

  Args:
    service_addr: Address of profiler service e.g. localhost:6009.
    logdir: Path of TensorBoard log directory e.g. /tmp/tb_log.
    duration_ms: Duration of tracing or monitoring in ms.
    worker_list: The list of worker TPUs that we are about to profile in the
      current session. (TPU only)
    include_dataset_ops: Set to false to profile longer traces.
    num_tracing_attempts: Automatically retry N times when no trace event is
      collected.

  Raises:
    UnavailableError: If no trace event is collected.
  """
    _pywrap_profiler.trace(service_addr, logdir, worker_list, include_dataset_ops, duration_ms, num_tracing_attempts, {})