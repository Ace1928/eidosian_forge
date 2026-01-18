import copy
import os
import pickle
import warnings
import numpy as np
def columnName(self, axis, col):
    ax = self._info[self._interpretAxis(axis)]
    return ax['cols'][col]['name']