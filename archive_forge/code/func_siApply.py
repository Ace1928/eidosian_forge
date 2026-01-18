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
def siApply(val, siprefix):
    """
    """
    n = SI_PREFIX_EXPONENTS[siprefix] if siprefix != '' else 0
    if n > 0:
        return val * 10 ** n
    elif n < 0:
        return val / 10 ** (-n)
    else:
        return val