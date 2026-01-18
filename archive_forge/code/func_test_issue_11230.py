from functools import reduce
import itertools
from operator import add
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import Inverse, MatAdd, MatMul, Transpose
from sympy.polys.rootoftools import CRootOf
from sympy.series.order import O
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import signsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.function import count_ops
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.functions.special.hyper import meijerg
from sympy.simplify import cse_main, cse_opts
from sympy.utilities.iterables import subsets
from sympy.testing.pytest import XFAIL, raises
from sympy.matrices import (MutableDenseMatrix, MutableSparseMatrix,
from sympy.matrices.expressions import MatrixSymbol
def test_issue_11230():
    a, b, f, k, l, i = symbols('a b f k l i')
    p = [a * b * f * k * l, a * i * k ** 2 * l, f * i * k ** 2 * l]
    R, C = cse(p)
    assert not any((i.is_Mul for a in C for i in a.args))
    from sympy.core.random import choice
    from sympy.core.function import expand_mul
    s = symbols('a:m')
    ex = [Mul(*[choice(s) for i in range(5)]) for i in range(7)]
    for p in subsets(ex, 3):
        p = list(p)
        R, C = cse(p)
        assert not any((i.is_Mul for a in C for i in a.args))
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        assert p == C
    ex = [Add(*[choice(s[:7]) for i in range(5)]) for i in range(7)]
    for p in subsets(ex, 3):
        p = list(p)
        R, C = cse(p)
        assert not any((i.is_Add for a in C for i in a.args))
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        assert p == [expand_mul(i) for i in C]