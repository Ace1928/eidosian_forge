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
def tauStep(dtau, v0, x0, t0, g):
    gamma = (1.0 - v0 ** 2) ** (-0.5)
    if g == 0:
        dt = dtau * gamma
    else:
        v0g = v0 * gamma
        dt = (np.sinh(dtau * g + np.arcsinh(v0g)) - v0g) / g
    v1, x1, t1 = Simulation.hypTStep(dt, v0, x0, t0, g)
    return (v1, x1, t0 + dt)