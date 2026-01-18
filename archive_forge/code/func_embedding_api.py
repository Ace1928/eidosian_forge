import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def embedding_api(self, csource, packed=False, pack=None):
    self._cdef(csource, packed=packed, pack=pack, dllexport=True)
    if self._embedding is None:
        self._embedding = ''