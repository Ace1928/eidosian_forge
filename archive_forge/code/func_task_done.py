import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
def task_done(self):
    with self._cond:
        if not self._unfinished_tasks.acquire(False):
            raise ValueError('task_done() called too many times')
        if self._unfinished_tasks._semlock._is_zero():
            self._cond.notify_all()