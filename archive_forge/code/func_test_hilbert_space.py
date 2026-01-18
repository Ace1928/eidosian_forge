from sympy.physics.quantum.hilbert import (
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.printing.repr import srepr
from sympy.printing.str import sstr
from sympy.sets.sets import Interval
def test_hilbert_space():
    hs = HilbertSpace()
    assert isinstance(hs, HilbertSpace)
    assert sstr(hs) == 'H'
    assert srepr(hs) == 'HilbertSpace()'