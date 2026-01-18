from sympy.physics.quantum.hilbert import (
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.printing.repr import srepr
from sympy.printing.str import sstr
from sympy.sets.sets import Interval
def test_direct_sum():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)
    h = hs1 + hs2
    assert isinstance(h, DirectSumHilbertSpace)
    assert h.dimension == 2 + n
    assert h.spaces == (hs1, hs2)
    f = FockSpace()
    h = hs1 + f + hs2
    assert h.dimension is oo
    assert h.spaces == (hs1, f, hs2)