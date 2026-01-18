import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def new_handle(self, x):
    return self._backend.newp_handle(self.BVoidP, x)