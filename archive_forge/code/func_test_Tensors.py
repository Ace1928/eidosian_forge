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
def test_Tensors():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p q r s')
    AT = AntiSymmetricTensor
    assert AT('t', (a, b), (i, j)) == -AT('t', (b, a), (i, j))
    assert AT('t', (a, b), (i, j)) == AT('t', (b, a), (j, i))
    assert AT('t', (a, b), (i, j)) == -AT('t', (a, b), (j, i))
    assert AT('t', (a, a), (i, j)) == 0
    assert AT('t', (a, b), (i, i)) == 0
    assert AT('t', (a, b, c), (i, j)) == -AT('t', (b, a, c), (i, j))
    assert AT('t', (a, b, c), (i, j, k)) == AT('t', (b, a, c), (i, k, j))
    tabij = AT('t', (a, b), (i, j))
    assert tabij.has(a)
    assert tabij.has(b)
    assert tabij.has(i)
    assert tabij.has(j)
    assert tabij.subs(b, c) == AT('t', (a, c), (i, j))
    assert (2 * tabij).subs(i, c) == 2 * AT('t', (a, b), (c, j))
    assert tabij.symbol == Symbol('t')
    assert latex(tabij) == '{t^{ab}_{ij}}'
    assert str(tabij) == 't((_a, _b),(_i, _j))'
    assert AT('t', (a, a), (i, j)).subs(a, b) == AT('t', (b, b), (i, j))
    assert AT('t', (a, i), (a, j)).subs(a, b) == AT('t', (b, i), (b, j))