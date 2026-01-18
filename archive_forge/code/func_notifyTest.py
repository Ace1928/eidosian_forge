import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
def notifyTest(self, *args):
    new_args = []
    for arg in args:
        new_args.append(_encode_if_needed(arg))
    args = tuple(new_args)
    self.notifications_queue.put_nowait(ParallelNotification('notifyTest', args))