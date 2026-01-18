import math
import pytest
from mpmath import *
def test_coulomb():
    mp.dps = 15
    assert coulombg(mpc(-5, 0), 2, 3).ae(20.08772948772143)