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
def cancel_join_thread(self):
    debug('Queue.cancel_join_thread()')
    self._joincancelled = True
    try:
        self._jointhread.cancel()
    except AttributeError:
        pass