import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def stepTo(self, t):
    data = self.clock.refData
    while self.i < len(data) - 1 and data['t'][self.i] < t:
        self.i += 1
    while self.i > 1 and data['t'][self.i - 1] >= t:
        self.i -= 1
    self.setPos(data['x'][self.i], self.clock.y0)
    t = data['pt'][self.i]
    self.hand.setRotation(-0.25 * t * 360.0)
    v = data['v'][self.i]
    gam = (1.0 - v ** 2) ** 0.5
    self.setTransform(QtGui.QTransform.fromScale(gam, 1.0))
    f = data['f'][self.i]
    tr = QtGui.QTransform()
    if f < 0:
        tr.translate(self.size * 0.4, 0)
    else:
        tr.translate(-self.size * 0.4, 0)
    tr.scale(-f * (0.5 + np.random.random() * 0.1), 1.0)
    self.flare.setTransform(tr)
    if self._spaceline is not None:
        self._spaceline.setPos(pg.Point(data['x'][self.i], data['t'][self.i]))
        self._spaceline.setAngle(data['v'][self.i] * 45.0)