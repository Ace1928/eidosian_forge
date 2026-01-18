from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, Lambda, diff, Function
from sympy.core.numbers import (zoo, Float, Integer, I, oo, pi, E,
from sympy.core.relational import Lt, Ge, Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (factorial2,
from sympy.functions.combinatorial.numbers import (lucas, bell,
from sympy.functions.elementary.complexes import re, im, conjugate, Abs
from sympy.functions.elementary.exponential import exp, LambertW, log
from sympy.functions.elementary.hyperbolic import (tanh, acoth, atanh,
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import (csc, sec, tan,
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.elliptic_integrals import (elliptic_pi,
from sympy.functions.special.error_functions import (fresnelc,
from sympy.functions.special.gamma_functions import (gamma, uppergamma,
from sympy.functions.special.mathieu_functions import (mathieusprime,
from sympy.functions.special.polynomials import (jacobi, chebyshevu,
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import (polylog, stieltjes,
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Xor, Or, false, true, And, Equivalent,
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.ntheory.factor_ import (totient, reduced_totient, primenu,
from sympy.physics.quantum import (ComplexSpace, FockSpace, hbar,
from sympy.printing.mathml import (MathMLPresentationPrinter,
from sympy.series.limits import Limit
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, Union, SymmetricDifference,
from sympy.stats.rv import RandomSymbol
from sympy.tensor.indexed import IndexedBase
from sympy.vector import (Divergence, CoordSys3D, Cross, Curl, Dot,
from sympy.testing.pytest import raises
def test_content_mathml_trig():
    mml = mp._print(sin(x))
    assert mml.childNodes[0].nodeName == 'sin'
    mml = mp._print(cos(x))
    assert mml.childNodes[0].nodeName == 'cos'
    mml = mp._print(tan(x))
    assert mml.childNodes[0].nodeName == 'tan'
    mml = mp._print(cot(x))
    assert mml.childNodes[0].nodeName == 'cot'
    mml = mp._print(csc(x))
    assert mml.childNodes[0].nodeName == 'csc'
    mml = mp._print(sec(x))
    assert mml.childNodes[0].nodeName == 'sec'
    mml = mp._print(asin(x))
    assert mml.childNodes[0].nodeName == 'arcsin'
    mml = mp._print(acos(x))
    assert mml.childNodes[0].nodeName == 'arccos'
    mml = mp._print(atan(x))
    assert mml.childNodes[0].nodeName == 'arctan'
    mml = mp._print(acot(x))
    assert mml.childNodes[0].nodeName == 'arccot'
    mml = mp._print(acsc(x))
    assert mml.childNodes[0].nodeName == 'arccsc'
    mml = mp._print(asec(x))
    assert mml.childNodes[0].nodeName == 'arcsec'
    mml = mp._print(sinh(x))
    assert mml.childNodes[0].nodeName == 'sinh'
    mml = mp._print(cosh(x))
    assert mml.childNodes[0].nodeName == 'cosh'
    mml = mp._print(tanh(x))
    assert mml.childNodes[0].nodeName == 'tanh'
    mml = mp._print(coth(x))
    assert mml.childNodes[0].nodeName == 'coth'
    mml = mp._print(csch(x))
    assert mml.childNodes[0].nodeName == 'csch'
    mml = mp._print(sech(x))
    assert mml.childNodes[0].nodeName == 'sech'
    mml = mp._print(asinh(x))
    assert mml.childNodes[0].nodeName == 'arcsinh'
    mml = mp._print(atanh(x))
    assert mml.childNodes[0].nodeName == 'arctanh'
    mml = mp._print(acosh(x))
    assert mml.childNodes[0].nodeName == 'arccosh'
    mml = mp._print(acoth(x))
    assert mml.childNodes[0].nodeName == 'arccoth'
    mml = mp._print(acsch(x))
    assert mml.childNodes[0].nodeName == 'arccsch'
    mml = mp._print(asech(x))
    assert mml.childNodes[0].nodeName == 'arcsech'