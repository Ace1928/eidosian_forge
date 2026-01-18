from math import atan2, degrees
import numpy as np
from . import SRTTransform
from .Qt import QtGui
from .Transform3D import Transform3D
from .Vector import Vector
def as2D(self):
    """Return a QTransform representing the x,y portion of this transform (if possible)"""
    return SRTTransform.SRTTransform(self)