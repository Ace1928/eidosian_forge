from math import atan2, degrees
import numpy as np
from . import SRTTransform
from .Qt import QtGui
from .Transform3D import Transform3D
from .Vector import Vector
def getRotation(self):
    """Return (angle, axis) of rotation"""
    return (self._state['angle'], Vector(self._state['axis']))