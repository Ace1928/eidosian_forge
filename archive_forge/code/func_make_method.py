import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def make_method(name):

    def method(self, *args, **kwargs):
        CLASS = cp.ndarray if self._supports_cupy else self._numpy_array.__class__
        _method = getattr(CLASS, name)
        args = (self,) + args
        if self._supports_cupy:
            return _call_cupy(_method, args, kwargs)
        return _call_numpy(_method, args, kwargs)
    method.__doc__ = getattr(np.ndarray, name).__doc__
    return method