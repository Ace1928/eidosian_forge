import pytest
from mpmath import *
def test_differint():
    mp.dps = 15
    assert differint(lambda t: t, 2, -0.5).ae(8 * sqrt(2 / pi) / 3)