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
def raiseContextMenu(self, ev):
    menu = self.getMenu(ev)
    if menu is not None:
        self.scene().addParentContextMenus(self, menu, ev)
        menu.popup(ev.screenPos().toPoint())