from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def solve3DTransform(points1, points2):
    """
    Find a 3D transformation matrix that maps points1 onto points2.
    Points must be specified as either lists of 4 Vectors or 
    (4, 3) arrays.
    """
    import numpy.linalg
    pts = []
    for inp in (points1, points2):
        if isinstance(inp, np.ndarray):
            A = np.empty((4, 4), dtype=float)
            A[:, :3] = inp[:, :3]
            A[:, 3] = 1.0
        else:
            A = np.array([[inp[i].x(), inp[i].y(), inp[i].z(), 1] for i in range(4)])
        pts.append(A)
    matrix = np.zeros((4, 4))
    for i in range(3):
        matrix[i] = numpy.linalg.solve(pts[0], pts[1][:, i])
    return matrix