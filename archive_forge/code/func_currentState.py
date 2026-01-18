import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def currentState(self, relativeTo=None):
    pos = self['start']
    dir = self['dir']
    if relativeTo is None:
        return (pos, dir)
    else:
        trans = self.itemTransform(relativeTo)[0]
        p1 = trans.map(pos)
        p2 = trans.map(pos + dir)
        return (Point(p1), Point(p2 - p1))