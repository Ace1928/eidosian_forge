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
def stepAnimation(self):
    now = perf_counter()
    dt = (now - self.lastAnimTime) * self.params['Animation Speed']
    self.lastAnimTime = now
    self.animTime += dt
    if self.animTime > self.params['Duration']:
        self.animTime = 0
        for a in self.animations:
            a.restart()
    for a in self.animations:
        a.stepTo(self.animTime)