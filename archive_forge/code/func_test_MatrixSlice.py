from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core import (EulerGamma, GoldenRatio, Catalan)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.special.bessel import (airyai, airyaiprime, airybi, airybiprime)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import (fresnelc, fresnels)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import dirichlet_eta
from sympy.geometry.line import (Ray, Segment)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand, Nor, Not, Or, Xor)
from sympy.matrices.dense import (Matrix, diag)
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.trace import Trace
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
from sympy.polys.orderings import (grlex, ilex)
from sympy.polys.polytools import groebner
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Complement, FiniteSet, Intersection, Interval, Union)
from sympy.codegen.ast import (Assignment, AddAugmentedAssignment,
from sympy.core.expr import UnevaluatedExpr
from sympy.physics.quantum.trace import Tr
from sympy.functions import (Abs, Chi, Ci, Ei, KroneckerDelta,
from sympy.matrices import (Adjoint, Inverse, MatrixSymbol, Transpose,
from sympy.matrices.expressions import hadamard_power
from sympy.physics import mechanics
from sympy.physics.control.lti import (TransferFunction, Feedback, TransferFunctionMatrix,
from sympy.physics.units import joule, degree
from sympy.printing.pretty import pprint, pretty as xpretty
from sympy.printing.pretty.pretty_symbology import center_accent, is_combining
from sympy.sets.conditionset import ConditionSet
from sympy.sets import ImageSet, ProductSet
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
from sympy.tensor.array import (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
from sympy.tensor.functions import TensorProduct
from sympy.tensor.tensor import (TensorIndexType, tensor_indices, TensorHead,
from sympy.testing.pytest import raises, _both_exp_pow, warns_deprecated_sympy
from sympy.vector import CoordSys3D, Gradient, Curl, Divergence, Dot, Cross, Laplacian
import sympy as sym
def test_MatrixSlice():
    n = Symbol('n', integer=True)
    x, y, z, w, t = symbols('x y z w t')
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)
    expr = MatrixSlice(X, (None, None, None), (None, None, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = X[x:x + 1, y:y + 1]
    assert pretty(expr) == upretty(expr) == 'X[x:x + 1, y:y + 1]'
    expr = X[x:x + 1:2, y:y + 1:2]
    assert pretty(expr) == upretty(expr) == 'X[x:x + 1:2, y:y + 1:2]'
    expr = X[:x, y:]
    assert pretty(expr) == upretty(expr) == 'X[:x, y:]'
    expr = X[:x, y:]
    assert pretty(expr) == upretty(expr) == 'X[:x, y:]'
    expr = X[x:, :y]
    assert pretty(expr) == upretty(expr) == 'X[x:, :y]'
    expr = X[x:y, z:w]
    assert pretty(expr) == upretty(expr) == 'X[x:y, z:w]'
    expr = X[x:y:t, w:t:x]
    assert pretty(expr) == upretty(expr) == 'X[x:y:t, w:t:x]'
    expr = X[x::y, t::w]
    assert pretty(expr) == upretty(expr) == 'X[x::y, t::w]'
    expr = X[:x:y, :t:w]
    assert pretty(expr) == upretty(expr) == 'X[:x:y, :t:w]'
    expr = X[::x, ::y]
    assert pretty(expr) == upretty(expr) == 'X[::x, ::y]'
    expr = MatrixSlice(X, (0, None, None), (0, None, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = MatrixSlice(X, (None, n, None), (None, n, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = MatrixSlice(X, (0, n, None), (0, n, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = MatrixSlice(X, (0, n, 2), (0, n, 2))
    assert pretty(expr) == upretty(expr) == 'X[::2, ::2]'
    expr = X[1:2:3, 4:5:6]
    assert pretty(expr) == upretty(expr) == 'X[1:2:3, 4:5:6]'
    expr = X[1:3:5, 4:6:8]
    assert pretty(expr) == upretty(expr) == 'X[1:3:5, 4:6:8]'
    expr = X[1:10:2]
    assert pretty(expr) == upretty(expr) == 'X[1:10:2, :]'
    expr = Y[:5, 1:9:2]
    assert pretty(expr) == upretty(expr) == 'Y[:5, 1:9:2]'
    expr = Y[:5, 1:10:2]
    assert pretty(expr) == upretty(expr) == 'Y[:5, 1::2]'
    expr = Y[5, :5:2]
    assert pretty(expr) == upretty(expr) == 'Y[5:6, :5:2]'
    expr = X[0:1, 0:1]
    assert pretty(expr) == upretty(expr) == 'X[:1, :1]'
    expr = X[0:1:2, 0:1:2]
    assert pretty(expr) == upretty(expr) == 'X[:1:2, :1:2]'
    expr = (Y + Z)[2:, 2:]
    assert pretty(expr) == upretty(expr) == '(Y + Z)[2:, 2:]'