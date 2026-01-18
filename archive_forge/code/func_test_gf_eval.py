from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_eval():
    assert gf_eval([], 4, 11, ZZ) == 0
    assert gf_eval([], 27, 11, ZZ) == 0
    assert gf_eval([7], 4, 11, ZZ) == 7
    assert gf_eval([7], 27, 11, ZZ) == 7
    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 0, 11, ZZ) == 0
    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 4, 11, ZZ) == 9
    assert gf_eval([1, 0, 3, 2, 4, 3, 1, 2, 0], 27, 11, ZZ) == 5
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 0, 11, ZZ) == 5
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 4, 11, ZZ) == 3
    assert gf_eval([4, 0, 0, 4, 6, 0, 1, 3, 5], 27, 11, ZZ) == 9
    assert gf_multi_eval([3, 2, 1], [0, 1, 2, 3], 11, ZZ) == [1, 6, 6, 1]