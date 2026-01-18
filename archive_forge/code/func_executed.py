import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
@property
def executed(self):
    """How many submissions were executed (failed or not).

        :returns: how many submissions were executed
        :rtype: number
        """
    return self._executed