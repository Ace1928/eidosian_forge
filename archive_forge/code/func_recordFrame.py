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
def recordFrame(self, i):
    f = self.force()
    self.inertData[i] = (self.x, self.t, self.v, self.pt, self.m, f)
    self.refData[i] = (self.refx, self.reft, self.refv, self.pt, self.refm, f)