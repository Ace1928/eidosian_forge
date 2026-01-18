from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_complex_parts():
    assert fabs('3') == 3
    assert fabs(3 + 4j) == 5
    assert re(3) == 3
    assert re(1 + 4j) == 1
    assert im(3) == 0
    assert im(1 + 4j) == 4
    assert conj(3) == 3
    assert conj(3 + 4j) == 3 - 4j
    assert mpf(3).conjugate() == 3