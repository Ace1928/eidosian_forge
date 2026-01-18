import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def get_conjugate(self):
    """Conjugation of quaternion"""
    w, x, y, z = self._val
    result = _Quaternion.from_value(np.array((w, -x, -y, -z)))
    return result