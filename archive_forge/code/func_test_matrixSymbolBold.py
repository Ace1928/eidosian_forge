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
def test_matrixSymbolBold():

    def boldpretty(expr):
        return xpretty(expr, use_unicode=True, wrap_line=False, mat_symbol_style='bold')
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol('A', 2, 2)
    assert boldpretty(trace(A)) == 'tr(𝐀)'
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    C = MatrixSymbol('C', 3, 3)
    assert boldpretty(-A) == '-𝐀'
    assert boldpretty(A - A * B - B) == '-𝐁 -𝐀⋅𝐁 + 𝐀'
    assert boldpretty(-A * B - A * B * C - B) == '-𝐁 -𝐀⋅𝐁 -𝐀⋅𝐁⋅𝐂'
    A = MatrixSymbol('Addot', 3, 3)
    assert boldpretty(A) == '𝐀̈'
    omega = MatrixSymbol('omega', 3, 3)
    assert boldpretty(omega) == 'ω'
    omega = MatrixSymbol('omeganorm', 3, 3)
    assert boldpretty(omega) == '‖ω‖'
    a = Symbol('alpha')
    b = Symbol('b')
    c = MatrixSymbol('c', 3, 1)
    d = MatrixSymbol('d', 3, 1)
    assert boldpretty(a * B * c + b * d) == 'b⋅𝐝 + α⋅𝐁⋅𝐜'
    d = MatrixSymbol('delta', 3, 1)
    B = MatrixSymbol('Beta', 3, 3)
    assert boldpretty(a * B * c + b * d) == 'b⋅δ + α⋅Β⋅𝐜'
    A = MatrixSymbol('A_2', 3, 3)
    assert boldpretty(A) == '𝐀₂'