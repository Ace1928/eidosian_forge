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
def resetBrush(self):
    """Remove the brush set for this spot; the scatter plot's default brush will be used instead."""
    self._data['brush'] = None
    self.updateItem()