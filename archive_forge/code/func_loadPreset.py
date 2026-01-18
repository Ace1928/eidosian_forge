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
def loadPreset(self, param, preset):
    if preset == '':
        return
    path = os.path.abspath(os.path.dirname(__file__))
    fn = os.path.join(path, 'presets', preset + '.cfg')
    state = configfile.readConfigFile(fn)
    self.loadState(state)