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
def test_cse__performance():
    nexprs, nterms = (3, 20)
    x = symbols('x:%d' % nterms)
    exprs = [reduce(add, [x[j] * (-1) ** (i + j) for j in range(nterms)]) for i in range(nexprs)]
    assert (exprs[0] + exprs[1]).simplify() == 0
    subst, red = cse(exprs)
    assert len(subst) > 0, 'exprs[0] == -exprs[2], i.e. a CSE'
    for i, e in enumerate(red):
        assert (e.subs(reversed(subst)) - exprs[i]).simplify() == 0