import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def get_axisangle(self):
    """Returns angle and vector of quaternion"""
    w, v = (self._val[0], self._val[1:])
    theta = acos(w) * 2.0
    return (theta, _normalize(v))