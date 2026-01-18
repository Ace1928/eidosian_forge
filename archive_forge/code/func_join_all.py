import multiprocessing
import requests
from . import thread
from .._compat import queue
def join_all(self):
    """Join all the threads to the master thread."""
    for session_thread in self._pool:
        session_thread.join()