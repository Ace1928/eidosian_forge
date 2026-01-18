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
def mouseDragEvent(self, ev, axis=None):
    ev.accept()
    pos = ev.pos()
    lastPos = ev.lastPos()
    dif = pos - lastPos
    dif = dif * -1
    mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float64)
    mask = mouseEnabled.copy()
    if axis is not None:
        mask[1 - axis] = 0.0
    if ev.button() in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton]:
        if self.state['mouseMode'] == ViewBox.RectMode and axis is None:
            if ev.isFinish():
                self.rbScaleBox.hide()
                ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                ax = self.childGroup.mapRectFromParent(ax)
                self.showAxRect(ax)
                self.axHistoryPointer += 1
                self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
            else:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        else:
            tr = self.childGroup.transform()
            tr = fn.invertQTransform(tr)
            tr = tr.map(dif * mask) - tr.map(Point(0, 0))
            x = tr.x() if mask[0] == 1 else None
            y = tr.y() if mask[1] == 1 else None
            self._resetTarget()
            if x is not None or y is not None:
                self.translateBy(x=x, y=y)
            self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
    elif ev.button() & QtCore.Qt.MouseButton.RightButton:
        if self.state['aspectLocked'] is not False:
            mask[0] = 0
        dif = ev.screenPos() - ev.lastScreenPos()
        dif = np.array([dif.x(), dif.y()])
        dif[0] *= -1
        s = (mask * 0.02 + 1) ** dif
        tr = self.childGroup.transform()
        tr = fn.invertQTransform(tr)
        x = s[0] if mouseEnabled[0] == 1 else None
        y = s[1] if mouseEnabled[1] == 1 else None
        center = Point(tr.map(ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton)))
        self._resetTarget()
        self.scaleBy(x=x, y=y, center=center)
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])