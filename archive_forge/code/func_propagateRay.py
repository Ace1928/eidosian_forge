import csv
import gzip
import os
from math import asin, atan2, cos, degrees, hypot, sin, sqrt
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui
def propagateRay(self, ray):
    """Refract, reflect, absorb, and/or scatter ray. This function may create and return new rays"""
    surface = self.surfaces[0]
    p1, ai = surface.intersectRay(ray)
    if p1 is not None:
        p1 = surface.mapToItem(ray, p1)
        rd = ray['dir']
        a1 = atan2(rd[1], rd[0])
        ar = a1 + np.pi - 2 * ai
        ray.setEnd(p1)
        dp = Point(cos(ar), sin(ar))
        ray = Ray(parent=ray, dir=dp)
    else:
        ray.setEnd(None)
    return [ray]