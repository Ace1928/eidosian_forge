from math import atan2, degrees
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
def clearMarkers(self):
    """ Remove all markers from this line.
        """
    self.markers = []
    self._maxMarkerSize = 0
    self.update()