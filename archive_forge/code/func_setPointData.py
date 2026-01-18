import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def setPointData(self, data, dataSet=None, mask=None):
    if dataSet is None:
        dataSet = self.data
    if isinstance(data, np.ndarray) or isinstance(data, list):
        if mask is not None:
            data = data[mask]
        if len(data) != len(dataSet):
            raise Exception('Length of meta data does not match number of points (%d != %d)' % (len(data), len(dataSet)))
    if isinstance(data, np.ndarray) and data.dtype.fields is not None and (len(data.dtype.fields) > 1):
        for i, rec in enumerate(data):
            dataSet['data'][i] = rec
    else:
        dataSet['data'] = data