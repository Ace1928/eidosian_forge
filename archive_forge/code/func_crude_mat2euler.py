import math
import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import eulerangles as nea
from .. import quaternions as nq
def crude_mat2euler(M):
    """The simplest possible - ignoring atan2 instability"""
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    return (math.atan2(-r12, r11), math.asin(r13), math.atan2(-r23, r33))