import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def sympy_euler(z, y, x):
    cos = math.cos
    sin = math.sin
    return [[cos(y) * cos(z), -cos(y) * sin(z), sin(y)], [cos(x) * sin(z) + cos(z) * sin(x) * sin(y), cos(x) * cos(z) - sin(x) * sin(y) * sin(z), -cos(y) * sin(x)], [sin(x) * sin(z) - cos(x) * cos(z) * sin(y), cos(z) * sin(x) + cos(x) * sin(y) * sin(z), cos(x) * cos(y)]]