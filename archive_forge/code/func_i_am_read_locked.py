import threading
import fasteners
from fasteners import test
@fasteners.read_locked
def i_am_read_locked(self, cb):
    cb(self._lock.owner)