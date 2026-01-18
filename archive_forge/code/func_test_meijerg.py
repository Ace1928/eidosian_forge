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
def test_meijerg():
    expr = meijerg([pi, pi, x], [1], [0, 1], [1, 2, 3], z)
    ucode_str = '╭─╮2, 3 ⎛π, π, x     1    │  ⎞\n│╶┐     ⎜                 │ z⎟\n╰─╯4, 5 ⎝ 0, 1    1, 2, 3 │  ⎠'
    ascii_str = ' __2, 3 /pi, pi, x     1    |  \\\n/__     |                   | z|\n\\_|4, 5 \\  0, 1     1, 2, 3 |  /'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = meijerg([1, pi / 7], [2, pi, 5], [], [], z ** 2)
    ucode_str = '        ⎛   π          │   ⎞\n╭─╮0, 2 ⎜1, ─  2, π, 5 │  2⎟\n│╶┐     ⎜   7          │ z ⎟\n╰─╯5, 0 ⎜              │   ⎟\n        ⎝              │   ⎠'
    ascii_str = '        /   pi           |   \\\n __0, 2 |1, --  2, pi, 5 |  2|\n/__     |   7            | z |\n\\_|5, 0 |                |   |\n        \\                |   /'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    ucode_str = '╭─╮ 1, 10 ⎛1, 1, 1, 1, 1, 1, 1, 1, 1, 1  1 │  ⎞\n│╶┐       ⎜                                │ z⎟\n╰─╯11,  2 ⎝             1                1 │  ⎠'
    ascii_str = ' __ 1, 10 /1, 1, 1, 1, 1, 1, 1, 1, 1, 1  1 |  \\\n/__       |                                | z|\n\\_|11,  2 \\             1                1 |  /'
    expr = meijerg([1] * 10, [1], [1], [1], z)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = meijerg([1, 2], [4, 3], [3], [4, 5], 1 / (1 / (1 / (1 / x + 1) + 1) + 1))
    ucode_str = '        ⎛           │       1      ⎞\n        ⎜           │ ─────────────⎟\n        ⎜           │         1    ⎟\n╭─╮1, 2 ⎜1, 2  4, 3 │ 1 + ─────────⎟\n│╶┐     ⎜           │           1  ⎟\n╰─╯4, 3 ⎜ 3    4, 5 │     1 + ─────⎟\n        ⎜           │             1⎟\n        ⎜           │         1 + ─⎟\n        ⎝           │             x⎠'
    ascii_str = '        /           |       1      \\\n        |           | -------------|\n        |           |         1    |\n __1, 2 |1, 2  4, 3 | 1 + ---------|\n/__     |           |           1  |\n\\_|4, 3 | 3    4, 5 |     1 + -----|\n        |           |             1|\n        |           |         1 + -|\n        \\           |             x/'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Integral(expr, x)
    ucode_str = '⌠                                        \n⎮         ⎛           │       1      ⎞   \n⎮         ⎜           │ ─────────────⎟   \n⎮         ⎜           │         1    ⎟   \n⎮ ╭─╮1, 2 ⎜1, 2  4, 3 │ 1 + ─────────⎟   \n⎮ │╶┐     ⎜           │           1  ⎟ dx\n⎮ ╰─╯4, 3 ⎜ 3    4, 5 │     1 + ─────⎟   \n⎮         ⎜           │             1⎟   \n⎮         ⎜           │         1 + ─⎟   \n⎮         ⎝           │             x⎠   \n⌡                                        '
    ascii_str = '  /                                       \n |                                        \n |         /           |       1      \\   \n |         |           | -------------|   \n |         |           |         1    |   \n |  __1, 2 |1, 2  4, 3 | 1 + ---------|   \n | /__     |           |           1  | dx\n | \\_|4, 3 | 3    4, 5 |     1 + -----|   \n |         |           |             1|   \n |         |           |         1 + -|   \n |         \\           |             x/   \n |                                        \n/                                         '
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str