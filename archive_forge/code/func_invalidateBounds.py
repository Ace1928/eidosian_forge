from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def invalidateBounds(self):
    self._boundingRect = None
    self._boundsCache = [None, None]