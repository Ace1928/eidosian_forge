import multiprocessing
import requests
from . import thread
from .._compat import queue
def get_exception(self):
    """Get an exception from the pool.

        :rtype: :class:`~ThreadException`
        """
    try:
        request, exc = self._exc_queue.get_nowait()
    except queue.Empty:
        return None
    else:
        return ThreadException(request, exc)