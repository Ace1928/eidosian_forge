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
def resetPen(self):
    """Remove the pen set for this spot; the scatter plot's default pen will be used instead."""
    self._data['pen'] = None
    self.updateItem()