import math
import pytest
from mpmath import *
def test_incomplete_beta():
    mp.dps = 15
    assert betainc(-2, -3, 0.5, 0.75).ae(63.430567331125545)
    assert betainc(4.5, 0.5 + 2j, 2.5, 6).ae(0.26288011461306215 + 0.516256523446702j)
    assert betainc(4, 5, 0, 6).ae(90747.77142857143)