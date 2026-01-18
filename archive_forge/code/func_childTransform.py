import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def childTransform(self):
    """
        Return the transform that maps from child(item in the childGroup) coordinates to local coordinates.
        (This maps from inside the viewbox to outside)
        """
    self.updateMatrix()
    m = self.childGroup.transform()
    return m