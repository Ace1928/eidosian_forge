from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
from io import StringIO
def test_minimize():

    def key(x: int) -> int:
        return -x
    rl = minimize(inc, dec)
    assert rl(4) == 3
    rl = minimize(inc, dec, objective=key)
    assert rl(4) == 5