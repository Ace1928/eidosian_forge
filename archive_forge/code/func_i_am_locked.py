import threading
import fasteners
from fasteners import test
@fasteners.locked
def i_am_locked(self, cb):
    gotten = [lock.locked() for lock in self._lock]
    cb(gotten)