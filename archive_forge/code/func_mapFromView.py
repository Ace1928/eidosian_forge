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
def mapFromView(self, obj):
    """Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox"""
    self.updateMatrix()
    m = self.childTransform()
    return m.map(obj)