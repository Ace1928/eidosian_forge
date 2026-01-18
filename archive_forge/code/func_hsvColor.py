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
def hsvColor(hue, sat=1.0, val=1.0, alpha=1.0):
    """Generate a QColor from HSVa values. (all arguments are float 0.0-1.0)"""
    return QtGui.QColor.fromHsvF(hue, sat, val, alpha)