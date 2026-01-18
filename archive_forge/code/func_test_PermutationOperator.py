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
def test_PermutationOperator():
    p, q, r, s = symbols('p,q,r,s')
    f, g, h, i = map(Function, 'fghi')
    P = PermutationOperator
    assert P(p, q).get_permuted(f(p) * g(q)) == -f(q) * g(p)
    assert P(p, q).get_permuted(f(p, q)) == -f(q, p)
    assert P(p, q).get_permuted(f(p)) == f(p)
    expr = f(p) * g(q) * h(r) * i(s) - f(q) * g(p) * h(r) * i(s) - f(p) * g(q) * h(s) * i(r) + f(q) * g(p) * h(s) * i(r)
    perms = [P(p, q), P(r, s)]
    assert simplify_index_permutations(expr, perms) == P(p, q) * P(r, s) * f(p) * g(q) * h(r) * i(s)
    assert latex(P(p, q)) == 'P(pq)'