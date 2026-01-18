import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def roiChanged(self, *args):
    pos = self.roi.pos()
    self.gitem.resetTransform()
    self.gitem.setRotation(self.roi.angle())
    br = self.gitem.boundingRect()
    o1 = self.gitem.mapToParent(br.topLeft())
    self.setParams(angle=self.roi.angle(), pos=pos + (self.gitem.pos() - o1))