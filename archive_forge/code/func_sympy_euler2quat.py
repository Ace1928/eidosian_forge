import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def sympy_euler2quat(z=0, y=0, x=0):
    cos = math.cos
    sin = math.sin
    return (cos(0.5 * x) * cos(0.5 * y) * cos(0.5 * z) - sin(0.5 * x) * sin(0.5 * y) * sin(0.5 * z), cos(0.5 * x) * sin(0.5 * y) * sin(0.5 * z) + cos(0.5 * y) * cos(0.5 * z) * sin(0.5 * x), cos(0.5 * x) * cos(0.5 * z) * sin(0.5 * y) - cos(0.5 * y) * sin(0.5 * x) * sin(0.5 * z), cos(0.5 * x) * cos(0.5 * y) * sin(0.5 * z) + cos(0.5 * z) * sin(0.5 * x) * sin(0.5 * y))