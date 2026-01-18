import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def mkPath(self):
    self.prepareGeometryChange()
    self.path = QtGui.QPainterPath()
    self.path.moveTo(self['start'])
    if self['end'] is not None:
        self.path.lineTo(self['end'])
    else:
        self.path.lineTo(self['start'] + 500 * self['dir'])