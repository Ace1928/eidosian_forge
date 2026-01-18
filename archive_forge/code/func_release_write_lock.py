from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
def release_write_lock(self):
    self._release()