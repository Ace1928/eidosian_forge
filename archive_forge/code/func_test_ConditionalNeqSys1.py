from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_ConditionalNeqSys1():
    from math import pi, sin

    def f_a(x, p):
        return [sin(p[0] * x[0])]

    def f_b(x, p):
        return [x[0] * (p[1] - x[0])]

    def factory(conds):
        return NeqSys(1, 1, f_b) if conds[0] else NeqSys(1, 1, f_a)
    cneqsys = ConditionalNeqSys([(lambda x, p: x[0] > 0, lambda x, p: x[0] >= 0)], factory)
    x, sol = cneqsys.solve([0], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0]) < 1e-13
    x, sol = cneqsys.solve([-1.4], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0] + 1) < 1e-13
    x, sol = cneqsys.solve([2], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 3) < 1e-13
    x, sol = cneqsys.solve([7], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 3) < 1e-13