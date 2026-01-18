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
def linkView(self, axis, view):
    """
        Link X or Y axes of two views and unlink any previously connected axes. *axis* must be ViewBox.XAxis or ViewBox.YAxis.
        If view is None, the axis is left unlinked.
        """
    if isinstance(view, str):
        if view == '':
            view = None
        else:
            view = ViewBox.NamedViews.get(view, view)
    if hasattr(view, 'implements') and view.implements('ViewBoxWrapper'):
        view = view.getViewBox()
    if axis == ViewBox.XAxis:
        signal = 'sigXRangeChanged'
        slot = self.linkedXChanged
    else:
        signal = 'sigYRangeChanged'
        slot = self.linkedYChanged
    oldLink = self.linkedView(axis)
    if oldLink is not None:
        try:
            getattr(oldLink, signal).disconnect(slot)
            oldLink.sigResized.disconnect(slot)
        except (TypeError, RuntimeError):
            pass
    if view is None or isinstance(view, str):
        self.state['linkedViews'][axis] = view
    else:
        self.state['linkedViews'][axis] = weakref.ref(view)
        getattr(view, signal).connect(slot)
        view.sigResized.connect(slot)
        if view.autoRangeEnabled()[axis] is not False:
            self.enableAutoRange(axis, False)
            slot()
        elif self.autoRangeEnabled()[axis] is False:
            slot()
    self.sigStateChanged.emit(self)