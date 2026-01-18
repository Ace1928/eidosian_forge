import copy
import os
import pickle
import warnings
import numpy as np
def columnUnits(self, axis, column):
    """Return the units for column in axis"""
    ax = self._info[self._interpretAxis(axis)]
    if 'cols' in ax:
        for c in ax['cols']:
            if c['name'] == column:
                return c['units']
        raise Exception('Axis %s has no column named %s' % (str(axis), str(column)))
    else:
        raise Exception('Axis %s has no column definitions' % str(axis))