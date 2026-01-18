import math
import pytest
from mpmath import *
def test_agm():
    mp.dps = 15
    assert agm(0, 0) == 0
    assert agm(0, 1) == 0
    assert agm(1, 1) == 1
    assert agm(7, 7) == 7
    assert agm(j, j) == j
    assert (1 / agm(1, sqrt(2))).ae(0.8346268416740732)
    assert agm(1, 2).ae(1.4567910310469068)
    assert agm(1, 3).ae(1.8636167832448964)
    assert agm(1, j).ae(0.5990701173677961 + 0.5990701173677961j)
    assert agm(2) == agm(1, 2)
    assert agm(-3, 4).ae(0.6346850976655091 + 1.3443087080896272j)