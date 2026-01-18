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
def mapFromViewToItem(self, item, obj):
    """Maps *obj* from view coordinates to the local coordinate system of *item*."""
    self.updateMatrix()
    return self.childGroup.mapToItem(item, obj)