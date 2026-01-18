from sympy.physics.secondquant import (
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex
def test_canonical_ordering_AntiSymmetricTensor():
    v = symbols('v')
    c, d = symbols(('c', 'd'), above_fermi=True, cls=Dummy)
    k, l = symbols(('k', 'l'), below_fermi=True, cls=Dummy)
    assert AntiSymmetricTensor(v, (k, l), (d, c)) == -AntiSymmetricTensor(v, (l, k), (d, c))