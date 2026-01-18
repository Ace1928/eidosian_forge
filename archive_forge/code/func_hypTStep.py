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
@staticmethod
def hypTStep(dt, v0, x0, tau0, g):
    if g == 0:
        return (v0, x0 + v0 * dt, tau0 + dt * (1.0 - v0 ** 2) ** 0.5)
    v02 = v0 ** 2
    g2 = g ** 2
    tinit = v0 / (g * (1 - v02) ** 0.5)
    B = (1 + g2 * (dt + tinit) ** 2) ** 0.5
    v1 = g * (dt + tinit) / B
    dtau = (np.arcsinh(g * (dt + tinit)) - np.arcsinh(g * tinit)) / g
    tau1 = tau0 + dtau
    x1 = x0 + 1.0 / g * (B - 1.0 / (1.0 - v02) ** 0.5)
    return (v1, x1, tau1)