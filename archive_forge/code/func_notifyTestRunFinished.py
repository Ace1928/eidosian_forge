import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
def notifyTestRunFinished(self, *args):
    self.notifications_queue.put_nowait(ParallelNotification('notifyTestRunFinished', args))