import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
@property
def average_runtime(self):
    """The average runtime of all submissions executed.

        :returns: average runtime of all submissions executed
        :rtype: number
        :raises: ZeroDivisionError when no executions have occurred.
        """
    return self._runtime / self._executed