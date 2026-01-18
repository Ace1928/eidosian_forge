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
def test_issue_7840():
    C393 = sympify('Piecewise((C391 - 1.65, C390 < 0.5), (Piecewise((C391 - 1.65,         C391 > 2.35), (C392, True)), True))')
    C391 = sympify('Piecewise((2.05*C390**(-1.03), C390 < 0.5), (2.5*C390**(-0.625), True))')
    C393 = C393.subs('C391', C391)
    sub = {}
    sub['C390'] = 0.703451854
    sub['C392'] = 1.01417794
    ss_answer = C393.subs(sub)
    substitutions, new_eqn = cse(C393)
    for pair in substitutions:
        sub[pair[0].name] = pair[1].subs(sub)
    cse_answer = new_eqn[0].subs(sub)
    assert ss_answer == cse_answer
    expr = sympify("Piecewise((Symbol('ON'), Equality(Symbol('mode'), Symbol('ON'))),         (Piecewise((Piecewise((Symbol('OFF'), StrictLessThan(Symbol('x'),         Symbol('threshold'))), (Symbol('ON'), true)), Equality(Symbol('mode'),         Symbol('AUTO'))), (Symbol('OFF'), true)), true))")
    substitutions, new_eqn = cse(expr)
    assert new_eqn[0] == expr
    assert len(substitutions) < 1