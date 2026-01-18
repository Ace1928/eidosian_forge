import os
import shutil
import sys
import signal
import warnings
import threading
from _multiprocessing import sem_unlink
from multiprocessing import util
from . import spawn
def maybe_unlink(self, name, rtype):
    """Decrement the refcount of a resource, and delete it if it hits 0"""
    self.ensure_running()
    self._send('MAYBE_UNLINK', name, rtype)