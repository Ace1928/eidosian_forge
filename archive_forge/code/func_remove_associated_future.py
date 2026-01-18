from concurrent import futures
from collections import namedtuple
import copy
import logging
import sys
import threading
from s3transfer.compat import MAXINT
from s3transfer.compat import six
from s3transfer.exceptions import CancelledError, TransferNotDoneError
from s3transfer.utils import FunctionContainer
from s3transfer.utils import TaskSemaphore
def remove_associated_future(self, future):
    """Removes a future's association to the TransferFuture"""
    with self._associated_futures_lock:
        self._associated_futures.remove(future)