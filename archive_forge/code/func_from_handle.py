import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def from_handle(self, x):
    return self._backend.from_handle(x)