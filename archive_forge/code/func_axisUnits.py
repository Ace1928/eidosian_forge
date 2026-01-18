import copy
import os
import pickle
import warnings
import numpy as np
def axisUnits(self, axis):
    """Return the units for axis"""
    ax = self._info[self._interpretAxis(axis)]
    if 'units' in ax:
        return ax['units']