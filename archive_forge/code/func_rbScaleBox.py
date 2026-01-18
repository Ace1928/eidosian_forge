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
@rbScaleBox.setter
def rbScaleBox(self, scaleBox):
    if self._rbScaleBox is not None:
        self.removeItem(self._rbScaleBox)
    self._rbScaleBox = scaleBox
    if scaleBox is None:
        return None
    scaleBox.setZValue(1000000000.0)
    scaleBox.hide()
    self.addItem(scaleBox, ignoreBounds=True)
    return None