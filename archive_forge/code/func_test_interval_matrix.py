from __future__ import division
import pytest
from mpmath import *
def test_interval_matrix():
    mp.dps = 15
    iv.dps = 15
    a = iv.matrix([['0.1', '0.3', '1.0'], ['7.1', '5.5', '4.8'], ['3.2', '4.4', '5.6']])
    b = iv.matrix(['4', '0.6', '0.5'])
    c = iv.lu_solve(a, b)
    assert c[0].delta < 1e-13
    assert c[1].delta < 1e-13
    assert c[2].delta < 1e-13
    assert 5.258232711306257 in c[0]
    assert -13.155049396267838 in c[1]
    assert 7.420691547749725 in c[2]