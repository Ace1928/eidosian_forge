from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import threading
import tensorflow as tf
def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Create threads to run the enqueue ops for the given session.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator, that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the caller must
        call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
    with self._lock:
        try:
            if self._runs_per_session[sess] > 0:
                return []
        except KeyError:
            pass
        self._runs_per_session[sess] = len(self._enqueue_ops)
        self._exceptions_raised = []
    ret_threads = [threading.Thread(target=self._run, args=(sess, op, feed_fn, coord)) for op, feed_fn in zip(self._enqueue_ops, self._feed_fns)]
    if coord:
        ret_threads.append(threading.Thread(target=self._close_on_stop, args=(sess, self._cancel_op, coord)))
    for t in ret_threads:
        if daemon:
            t.daemon = True
        if start:
            t.start()
    return ret_threads