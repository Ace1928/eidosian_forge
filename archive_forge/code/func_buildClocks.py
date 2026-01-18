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
def buildClocks(self):
    clocks = {}
    template = self.param('ClockTemplate')
    spacing = self['Spacing']
    for i in range(self['Number of Clocks']):
        c = list(template.buildClocks().values())[0]
        c.x0 += i * spacing
        clocks[self.name() + '%02d' % i] = c
    return clocks