import copy
import os
import pickle
import warnings
import numpy as np
def axisCollapsingFn(self, fn, axis=None, *args, **kargs):
    fn = getattr(self._data, fn)
    if axis is None:
        return fn(axis, *args, **kargs)
    else:
        info = self.infoCopy()
        axis = self._interpretAxis(axis)
        info.pop(axis)
        return MetaArray(fn(axis, *args, **kargs), info=info)