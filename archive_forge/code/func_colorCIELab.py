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
def colorCIELab(qcol):
    """
    Describes a QColor by an array of CIE L*a*b* values.
    Also see :func:`CIELabColor() <pyqtgraph.CIELabColor>` .

    Parameters
    ----------
    qcol: QColor
        QColor to be converted

    Returns
    -------
    np.ndarray 
        Color coordinates `[L, a, b]`.
    """
    srgb = qcol.getRgbF()[:3]
    vec_RGB = np.zeros(3)
    for idx, val in enumerate(srgb):
        if val > 12.92 * 0.0031308:
            vec_RGB[idx] = ((val + 0.055) / 1.055) ** 2.4
        else:
            vec_RGB[idx] = val / 12.92
    vec_XYZ = MATRIX_XYZ_FROM_RGB @ vec_RGB
    vec_XYZ1 = vec_XYZ / VECTOR_XYZn
    for idx, val in enumerate(vec_XYZ1):
        if val > 0.008856:
            vec_XYZ1[idx] = vec_XYZ1[idx] ** (1 / 3)
        else:
            vec_XYZ1[idx] = 7.787 * vec_XYZ1[idx] + 16 / 116
    vec_Lab = np.array([116 * vec_XYZ1[1] - 16, 500 * (vec_XYZ1[0] - vec_XYZ1[1]), 200 * (vec_XYZ1[1] - vec_XYZ1[2])])
    return vec_Lab