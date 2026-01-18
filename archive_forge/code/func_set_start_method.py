import os
import sys
import threading
from . import process
from . import reduction
def set_start_method(self, method, force=False):
    if self._actual_context is not None and (not force):
        raise RuntimeError('context has already been set')
    if method is None and force:
        self._actual_context = None
        return
    self._actual_context = self.get_context(method)