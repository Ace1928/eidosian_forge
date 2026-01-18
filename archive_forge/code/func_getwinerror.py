import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def getwinerror(self, code=-1):
    return self._backend.getwinerror(code)