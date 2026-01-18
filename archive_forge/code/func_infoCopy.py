import copy
import os
import pickle
import warnings
import numpy as np
def infoCopy(self, axis=None):
    """Return a deep copy of the axis meta info for this object"""
    if axis is None:
        return copy.deepcopy(self._info)
    else:
        return copy.deepcopy(self._info[self._interpretAxis(axis)])