import math
import pytest
from mpmath import *
def test_hypercomb_zero_pow():
    assert hypercomb(lambda a: (([0], [a], [], [], [], [], 0),), [0]) == 1
    assert meijerg([[-1.5], []], [[0], [-0.75]], 0).ae(1.4464090846320772)