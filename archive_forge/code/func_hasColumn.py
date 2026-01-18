import copy
import os
import pickle
import warnings
import numpy as np
def hasColumn(self, axis, col):
    ax = self._info[self._interpretAxis(axis)]
    if 'cols' in ax:
        for c in ax['cols']:
            if c['name'] == col:
                return True
    return False