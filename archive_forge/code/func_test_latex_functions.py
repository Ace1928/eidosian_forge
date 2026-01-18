from sympy import MatAdd, MatMul, Array
from sympy.algebras.quaternion import Quaternion
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.permutations import Cycle, Permutation, AppliedPermutation
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple, Dict
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import (Derivative, Function, Lambda, Subs, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (AlgebraicNumber, Float, I, Integer, Rational, oo, pi)
from sympy.core.parameters import evaluate
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial, factorial2, subfactorial)
from sympy.functions.combinatorial.numbers import bernoulli, bell, catalan, euler, genocchi, lucas, fibonacci, tribonacci
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, coth)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (Max, Min, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acsc, asin, cos, cot, sin, tan)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f, elliptic_k, elliptic_pi)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.spherical_harmonics import (Ynm, Znm)
from sympy.functions.special.tensor_functions import (KroneckerDelta, LeviCivita)
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, stieltjes, zeta)
from sympy.integrals.integrals import Integral
from sympy.integrals.transforms import (CosineTransform, FourierTransform, InverseCosineTransform, InverseFourierTransform, InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform, LaplaceTransform, MellinTransform, SineTransform)
from sympy.logic import Implies
from sympy.logic.boolalg import (And, Or, Xor, Equivalent, false, Not, true)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.kronecker import KroneckerProduct
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.permutation import PermutationMatrix
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.physics.control.lti import TransferFunction, Series, Parallel, Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.ntheory.factor_ import (divisor_sigma, primenu, primeomega, reduced_totient, totient, udivisor_sigma)
from sympy.physics.quantum import Commutator, Operator
from sympy.physics.quantum.trace import Tr
from sympy.physics.units import meter, gibibyte, gram, microgram, second, milli, micro
from sympy.polys.domains.integerring import ZZ
from sympy.polys.fields import field
from sympy.polys.polytools import Poly
from sympy.polys.rings import ring
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ComplexRegion, ImageSet, Range)
from sympy.sets.ordinals import Ordinal, OrdinalOmega, OmegaPower
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import (FiniteSet, Interval, Union, Intersection, Complement, SymmetricDifference, ProductSet)
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
from sympy.tensor.array import (ImmutableDenseNDimArray,
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.tensor.toperators import PartialDerivative
from sympy.vector import CoordSys3D, Cross, Curl, Dot, Divergence, Gradient, Laplacian
from sympy.testing.pytest import (XFAIL, raises, _both_exp_pow,
from sympy.printing.latex import (latex, translate, greek_letters_set,
import sympy as sym
from sympy.abc import mu, tau
@_both_exp_pow
def test_latex_functions():
    assert latex(exp(x)) == 'e^{x}'
    assert latex(exp(1) + exp(2)) == 'e + e^{2}'
    f = Function('f')
    assert latex(f(x)) == 'f{\\left(x \\right)}'
    assert latex(f) == 'f'
    g = Function('g')
    assert latex(g(x, y)) == 'g{\\left(x,y \\right)}'
    assert latex(g) == 'g'
    h = Function('h')
    assert latex(h(x, y, z)) == 'h{\\left(x,y,z \\right)}'
    assert latex(h) == 'h'
    Li = Function('Li')
    assert latex(Li) == '\\operatorname{Li}'
    assert latex(Li(x)) == '\\operatorname{Li}{\\left(x \\right)}'
    mybeta = Function('beta')
    assert latex(mybeta(x, y, z)) == '\\beta{\\left(x,y,z \\right)}'
    assert latex(beta(x, y)) == '\\operatorname{B}\\left(x, y\\right)'
    assert latex(beta(x, evaluate=False)) == '\\operatorname{B}\\left(x, x\\right)'
    assert latex(beta(x, y) ** 2) == '\\operatorname{B}^{2}\\left(x, y\\right)'
    assert latex(mybeta(x)) == '\\beta{\\left(x \\right)}'
    assert latex(mybeta) == '\\beta'
    g = Function('gamma')
    assert latex(g(x, y, z)) == '\\gamma{\\left(x,y,z \\right)}'
    assert latex(g(x)) == '\\gamma{\\left(x \\right)}'
    assert latex(g) == '\\gamma'
    a_1 = Function('a_1')
    assert latex(a_1) == 'a_{1}'
    assert latex(a_1(x)) == 'a_{1}{\\left(x \\right)}'
    assert latex(Function('a_1')) == 'a_{1}'
    assert latex(Function('ab')) == '\\operatorname{ab}'
    assert latex(Function('ab1')) == '\\operatorname{ab}_{1}'
    assert latex(Function('ab12')) == '\\operatorname{ab}_{12}'
    assert latex(Function('ab_1')) == '\\operatorname{ab}_{1}'
    assert latex(Function('ab_12')) == '\\operatorname{ab}_{12}'
    assert latex(Function('ab_c')) == '\\operatorname{ab}_{c}'
    assert latex(Function('ab_cd')) == '\\operatorname{ab}_{cd}'
    assert latex(Function('ab')(Symbol('x'))) == '\\operatorname{ab}{\\left(x \\right)}'
    assert latex(Function('ab1')(Symbol('x'))) == '\\operatorname{ab}_{1}{\\left(x \\right)}'
    assert latex(Function('ab12')(Symbol('x'))) == '\\operatorname{ab}_{12}{\\left(x \\right)}'
    assert latex(Function('ab_1')(Symbol('x'))) == '\\operatorname{ab}_{1}{\\left(x \\right)}'
    assert latex(Function('ab_c')(Symbol('x'))) == '\\operatorname{ab}_{c}{\\left(x \\right)}'
    assert latex(Function('ab_cd')(Symbol('x'))) == '\\operatorname{ab}_{cd}{\\left(x \\right)}'
    assert latex(Function('ab')() ** 2) == '\\operatorname{ab}^{2}{\\left( \\right)}'
    assert latex(Function('ab1')() ** 2) == '\\operatorname{ab}_{1}^{2}{\\left( \\right)}'
    assert latex(Function('ab12')() ** 2) == '\\operatorname{ab}_{12}^{2}{\\left( \\right)}'
    assert latex(Function('ab_1')() ** 2) == '\\operatorname{ab}_{1}^{2}{\\left( \\right)}'
    assert latex(Function('ab_12')() ** 2) == '\\operatorname{ab}_{12}^{2}{\\left( \\right)}'
    assert latex(Function('ab')(Symbol('x')) ** 2) == '\\operatorname{ab}^{2}{\\left(x \\right)}'
    assert latex(Function('ab1')(Symbol('x')) ** 2) == '\\operatorname{ab}_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('ab12')(Symbol('x')) ** 2) == '\\operatorname{ab}_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('ab_1')(Symbol('x')) ** 2) == '\\operatorname{ab}_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('ab_12')(Symbol('x')) ** 2) == '\\operatorname{ab}_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('a')) == 'a'
    assert latex(Function('a1')) == 'a_{1}'
    assert latex(Function('a12')) == 'a_{12}'
    assert latex(Function('a_1')) == 'a_{1}'
    assert latex(Function('a_12')) == 'a_{12}'
    assert latex(Function('a')()) == 'a{\\left( \\right)}'
    assert latex(Function('a1')()) == 'a_{1}{\\left( \\right)}'
    assert latex(Function('a12')()) == 'a_{12}{\\left( \\right)}'
    assert latex(Function('a_1')()) == 'a_{1}{\\left( \\right)}'
    assert latex(Function('a_12')()) == 'a_{12}{\\left( \\right)}'
    assert latex(Function('a')() ** 2) == 'a^{2}{\\left( \\right)}'
    assert latex(Function('a1')() ** 2) == 'a_{1}^{2}{\\left( \\right)}'
    assert latex(Function('a12')() ** 2) == 'a_{12}^{2}{\\left( \\right)}'
    assert latex(Function('a_1')() ** 2) == 'a_{1}^{2}{\\left( \\right)}'
    assert latex(Function('a_12')() ** 2) == 'a_{12}^{2}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** 2) == 'a^{2}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** 2) == 'a_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** 2) == 'a_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** 2) == 'a_{1}^{2}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** 2) == 'a_{12}^{2}{\\left(x \\right)}'
    assert latex(Function('a')() ** 32) == 'a^{32}{\\left( \\right)}'
    assert latex(Function('a1')() ** 32) == 'a_{1}^{32}{\\left( \\right)}'
    assert latex(Function('a12')() ** 32) == 'a_{12}^{32}{\\left( \\right)}'
    assert latex(Function('a_1')() ** 32) == 'a_{1}^{32}{\\left( \\right)}'
    assert latex(Function('a_12')() ** 32) == 'a_{12}^{32}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** 32) == 'a^{32}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** 32) == 'a_{1}^{32}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** 32) == 'a_{12}^{32}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** 32) == 'a_{1}^{32}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** 32) == 'a_{12}^{32}{\\left(x \\right)}'
    assert latex(Function('a')() ** a) == 'a^{a}{\\left( \\right)}'
    assert latex(Function('a1')() ** a) == 'a_{1}^{a}{\\left( \\right)}'
    assert latex(Function('a12')() ** a) == 'a_{12}^{a}{\\left( \\right)}'
    assert latex(Function('a_1')() ** a) == 'a_{1}^{a}{\\left( \\right)}'
    assert latex(Function('a_12')() ** a) == 'a_{12}^{a}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** a) == 'a^{a}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** a) == 'a_{1}^{a}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** a) == 'a_{12}^{a}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** a) == 'a_{1}^{a}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** a) == 'a_{12}^{a}{\\left(x \\right)}'
    ab = Symbol('ab')
    assert latex(Function('a')() ** ab) == 'a^{ab}{\\left( \\right)}'
    assert latex(Function('a1')() ** ab) == 'a_{1}^{ab}{\\left( \\right)}'
    assert latex(Function('a12')() ** ab) == 'a_{12}^{ab}{\\left( \\right)}'
    assert latex(Function('a_1')() ** ab) == 'a_{1}^{ab}{\\left( \\right)}'
    assert latex(Function('a_12')() ** ab) == 'a_{12}^{ab}{\\left( \\right)}'
    assert latex(Function('a')(Symbol('x')) ** ab) == 'a^{ab}{\\left(x \\right)}'
    assert latex(Function('a1')(Symbol('x')) ** ab) == 'a_{1}^{ab}{\\left(x \\right)}'
    assert latex(Function('a12')(Symbol('x')) ** ab) == 'a_{12}^{ab}{\\left(x \\right)}'
    assert latex(Function('a_1')(Symbol('x')) ** ab) == 'a_{1}^{ab}{\\left(x \\right)}'
    assert latex(Function('a_12')(Symbol('x')) ** ab) == 'a_{12}^{ab}{\\left(x \\right)}'
    assert latex(Function('a^12')(x)) == 'a^{12}{\\left(x \\right)}'
    assert latex(Function('a^12')(x) ** ab) == '\\left(a^{12}\\right)^{ab}{\\left(x \\right)}'
    assert latex(Function('a__12')(x)) == 'a^{12}{\\left(x \\right)}'
    assert latex(Function('a__12')(x) ** ab) == '\\left(a^{12}\\right)^{ab}{\\left(x \\right)}'
    assert latex(Function('a_1__1_2')(x)) == 'a^{1}_{1 2}{\\left(x \\right)}'
    omega1 = Function('omega1')
    assert latex(omega1) == '\\omega_{1}'
    assert latex(omega1(x)) == '\\omega_{1}{\\left(x \\right)}'
    assert latex(sin(x)) == '\\sin{\\left(x \\right)}'
    assert latex(sin(x), fold_func_brackets=True) == '\\sin {x}'
    assert latex(sin(2 * x ** 2), fold_func_brackets=True) == '\\sin {2 x^{2}}'
    assert latex(sin(x ** 2), fold_func_brackets=True) == '\\sin {x^{2}}'
    assert latex(asin(x) ** 2) == '\\operatorname{asin}^{2}{\\left(x \\right)}'
    assert latex(asin(x) ** 2, inv_trig_style='full') == '\\arcsin^{2}{\\left(x \\right)}'
    assert latex(asin(x) ** 2, inv_trig_style='power') == '\\sin^{-1}{\\left(x \\right)}^{2}'
    assert latex(asin(x ** 2), inv_trig_style='power', fold_func_brackets=True) == '\\sin^{-1} {x^{2}}'
    assert latex(acsc(x), inv_trig_style='full') == '\\operatorname{arccsc}{\\left(x \\right)}'
    assert latex(asinh(x), inv_trig_style='full') == '\\operatorname{arsinh}{\\left(x \\right)}'
    assert latex(factorial(k)) == 'k!'
    assert latex(factorial(-k)) == '\\left(- k\\right)!'
    assert latex(factorial(k) ** 2) == 'k!^{2}'
    assert latex(subfactorial(k)) == '!k'
    assert latex(subfactorial(-k)) == '!\\left(- k\\right)'
    assert latex(subfactorial(k) ** 2) == '\\left(!k\\right)^{2}'
    assert latex(factorial2(k)) == 'k!!'
    assert latex(factorial2(-k)) == '\\left(- k\\right)!!'
    assert latex(factorial2(k) ** 2) == 'k!!^{2}'
    assert latex(binomial(2, k)) == '{\\binom{2}{k}}'
    assert latex(binomial(2, k) ** 2) == '{\\binom{2}{k}}^{2}'
    assert latex(FallingFactorial(3, k)) == '{\\left(3\\right)}_{k}'
    assert latex(RisingFactorial(3, k)) == '{3}^{\\left(k\\right)}'
    assert latex(floor(x)) == '\\left\\lfloor{x}\\right\\rfloor'
    assert latex(ceiling(x)) == '\\left\\lceil{x}\\right\\rceil'
    assert latex(frac(x)) == '\\operatorname{frac}{\\left(x\\right)}'
    assert latex(floor(x) ** 2) == '\\left\\lfloor{x}\\right\\rfloor^{2}'
    assert latex(ceiling(x) ** 2) == '\\left\\lceil{x}\\right\\rceil^{2}'
    assert latex(frac(x) ** 2) == '\\operatorname{frac}{\\left(x\\right)}^{2}'
    assert latex(Min(x, 2, x ** 3)) == '\\min\\left(2, x, x^{3}\\right)'
    assert latex(Min(x, y) ** 2) == '\\min\\left(x, y\\right)^{2}'
    assert latex(Max(x, 2, x ** 3)) == '\\max\\left(2, x, x^{3}\\right)'
    assert latex(Max(x, y) ** 2) == '\\max\\left(x, y\\right)^{2}'
    assert latex(Abs(x)) == '\\left|{x}\\right|'
    assert latex(Abs(x) ** 2) == '\\left|{x}\\right|^{2}'
    assert latex(re(x)) == '\\operatorname{re}{\\left(x\\right)}'
    assert latex(re(x + y)) == '\\operatorname{re}{\\left(x\\right)} + \\operatorname{re}{\\left(y\\right)}'
    assert latex(im(x)) == '\\operatorname{im}{\\left(x\\right)}'
    assert latex(conjugate(x)) == '\\overline{x}'
    assert latex(conjugate(x) ** 2) == '\\overline{x}^{2}'
    assert latex(conjugate(x ** 2)) == '\\overline{x}^{2}'
    assert latex(gamma(x)) == '\\Gamma\\left(x\\right)'
    w = Wild('w')
    assert latex(gamma(w)) == '\\Gamma\\left(w\\right)'
    assert latex(Order(x)) == 'O\\left(x\\right)'
    assert latex(Order(x, x)) == 'O\\left(x\\right)'
    assert latex(Order(x, (x, 0))) == 'O\\left(x\\right)'
    assert latex(Order(x, (x, oo))) == 'O\\left(x; x\\rightarrow \\infty\\right)'
    assert latex(Order(x - y, (x, y))) == 'O\\left(x - y; x\\rightarrow y\\right)'
    assert latex(Order(x, x, y)) == 'O\\left(x; \\left( x, \\  y\\right)\\rightarrow \\left( 0, \\  0\\right)\\right)'
    assert latex(Order(x, x, y)) == 'O\\left(x; \\left( x, \\  y\\right)\\rightarrow \\left( 0, \\  0\\right)\\right)'
    assert latex(Order(x, (x, oo), (y, oo))) == 'O\\left(x; \\left( x, \\  y\\right)\\rightarrow \\left( \\infty, \\  \\infty\\right)\\right)'
    assert latex(lowergamma(x, y)) == '\\gamma\\left(x, y\\right)'
    assert latex(lowergamma(x, y) ** 2) == '\\gamma^{2}\\left(x, y\\right)'
    assert latex(uppergamma(x, y)) == '\\Gamma\\left(x, y\\right)'
    assert latex(uppergamma(x, y) ** 2) == '\\Gamma^{2}\\left(x, y\\right)'
    assert latex(cot(x)) == '\\cot{\\left(x \\right)}'
    assert latex(coth(x)) == '\\coth{\\left(x \\right)}'
    assert latex(re(x)) == '\\operatorname{re}{\\left(x\\right)}'
    assert latex(im(x)) == '\\operatorname{im}{\\left(x\\right)}'
    assert latex(root(x, y)) == 'x^{\\frac{1}{y}}'
    assert latex(arg(x)) == '\\arg{\\left(x \\right)}'
    assert latex(zeta(x)) == '\\zeta\\left(x\\right)'
    assert latex(zeta(x) ** 2) == '\\zeta^{2}\\left(x\\right)'
    assert latex(zeta(x, y)) == '\\zeta\\left(x, y\\right)'
    assert latex(zeta(x, y) ** 2) == '\\zeta^{2}\\left(x, y\\right)'
    assert latex(dirichlet_eta(x)) == '\\eta\\left(x\\right)'
    assert latex(dirichlet_eta(x) ** 2) == '\\eta^{2}\\left(x\\right)'
    assert latex(polylog(x, y)) == '\\operatorname{Li}_{x}\\left(y\\right)'
    assert latex(polylog(x, y) ** 2) == '\\operatorname{Li}_{x}^{2}\\left(y\\right)'
    assert latex(lerchphi(x, y, n)) == '\\Phi\\left(x, y, n\\right)'
    assert latex(lerchphi(x, y, n) ** 2) == '\\Phi^{2}\\left(x, y, n\\right)'
    assert latex(stieltjes(x)) == '\\gamma_{x}'
    assert latex(stieltjes(x) ** 2) == '\\gamma_{x}^{2}'
    assert latex(stieltjes(x, y)) == '\\gamma_{x}\\left(y\\right)'
    assert latex(stieltjes(x, y) ** 2) == '\\gamma_{x}\\left(y\\right)^{2}'
    assert latex(elliptic_k(z)) == 'K\\left(z\\right)'
    assert latex(elliptic_k(z) ** 2) == 'K^{2}\\left(z\\right)'
    assert latex(elliptic_f(x, y)) == 'F\\left(x\\middle| y\\right)'
    assert latex(elliptic_f(x, y) ** 2) == 'F^{2}\\left(x\\middle| y\\right)'
    assert latex(elliptic_e(x, y)) == 'E\\left(x\\middle| y\\right)'
    assert latex(elliptic_e(x, y) ** 2) == 'E^{2}\\left(x\\middle| y\\right)'
    assert latex(elliptic_e(z)) == 'E\\left(z\\right)'
    assert latex(elliptic_e(z) ** 2) == 'E^{2}\\left(z\\right)'
    assert latex(elliptic_pi(x, y, z)) == '\\Pi\\left(x; y\\middle| z\\right)'
    assert latex(elliptic_pi(x, y, z) ** 2) == '\\Pi^{2}\\left(x; y\\middle| z\\right)'
    assert latex(elliptic_pi(x, y)) == '\\Pi\\left(x\\middle| y\\right)'
    assert latex(elliptic_pi(x, y) ** 2) == '\\Pi^{2}\\left(x\\middle| y\\right)'
    assert latex(Ei(x)) == '\\operatorname{Ei}{\\left(x \\right)}'
    assert latex(Ei(x) ** 2) == '\\operatorname{Ei}^{2}{\\left(x \\right)}'
    assert latex(expint(x, y)) == '\\operatorname{E}_{x}\\left(y\\right)'
    assert latex(expint(x, y) ** 2) == '\\operatorname{E}_{x}^{2}\\left(y\\right)'
    assert latex(Shi(x) ** 2) == '\\operatorname{Shi}^{2}{\\left(x \\right)}'
    assert latex(Si(x) ** 2) == '\\operatorname{Si}^{2}{\\left(x \\right)}'
    assert latex(Ci(x) ** 2) == '\\operatorname{Ci}^{2}{\\left(x \\right)}'
    assert latex(Chi(x) ** 2) == '\\operatorname{Chi}^{2}\\left(x\\right)'
    assert latex(Chi(x)) == '\\operatorname{Chi}\\left(x\\right)'
    assert latex(jacobi(n, a, b, x)) == 'P_{n}^{\\left(a,b\\right)}\\left(x\\right)'
    assert latex(jacobi(n, a, b, x) ** 2) == '\\left(P_{n}^{\\left(a,b\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(gegenbauer(n, a, x)) == 'C_{n}^{\\left(a\\right)}\\left(x\\right)'
    assert latex(gegenbauer(n, a, x) ** 2) == '\\left(C_{n}^{\\left(a\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(chebyshevt(n, x)) == 'T_{n}\\left(x\\right)'
    assert latex(chebyshevt(n, x) ** 2) == '\\left(T_{n}\\left(x\\right)\\right)^{2}'
    assert latex(chebyshevu(n, x)) == 'U_{n}\\left(x\\right)'
    assert latex(chebyshevu(n, x) ** 2) == '\\left(U_{n}\\left(x\\right)\\right)^{2}'
    assert latex(legendre(n, x)) == 'P_{n}\\left(x\\right)'
    assert latex(legendre(n, x) ** 2) == '\\left(P_{n}\\left(x\\right)\\right)^{2}'
    assert latex(assoc_legendre(n, a, x)) == 'P_{n}^{\\left(a\\right)}\\left(x\\right)'
    assert latex(assoc_legendre(n, a, x) ** 2) == '\\left(P_{n}^{\\left(a\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(laguerre(n, x)) == 'L_{n}\\left(x\\right)'
    assert latex(laguerre(n, x) ** 2) == '\\left(L_{n}\\left(x\\right)\\right)^{2}'
    assert latex(assoc_laguerre(n, a, x)) == 'L_{n}^{\\left(a\\right)}\\left(x\\right)'
    assert latex(assoc_laguerre(n, a, x) ** 2) == '\\left(L_{n}^{\\left(a\\right)}\\left(x\\right)\\right)^{2}'
    assert latex(hermite(n, x)) == 'H_{n}\\left(x\\right)'
    assert latex(hermite(n, x) ** 2) == '\\left(H_{n}\\left(x\\right)\\right)^{2}'
    theta = Symbol('theta', real=True)
    phi = Symbol('phi', real=True)
    assert latex(Ynm(n, m, theta, phi)) == 'Y_{n}^{m}\\left(\\theta,\\phi\\right)'
    assert latex(Ynm(n, m, theta, phi) ** 3) == '\\left(Y_{n}^{m}\\left(\\theta,\\phi\\right)\\right)^{3}'
    assert latex(Znm(n, m, theta, phi)) == 'Z_{n}^{m}\\left(\\theta,\\phi\\right)'
    assert latex(Znm(n, m, theta, phi) ** 3) == '\\left(Z_{n}^{m}\\left(\\theta,\\phi\\right)\\right)^{3}'
    assert latex(polar_lift(0)) == '\\operatorname{polar\\_lift}{\\left(0 \\right)}'
    assert latex(polar_lift(0) ** 3) == '\\operatorname{polar\\_lift}^{3}{\\left(0 \\right)}'
    assert latex(totient(n)) == '\\phi\\left(n\\right)'
    assert latex(totient(n) ** 2) == '\\left(\\phi\\left(n\\right)\\right)^{2}'
    assert latex(reduced_totient(n)) == '\\lambda\\left(n\\right)'
    assert latex(reduced_totient(n) ** 2) == '\\left(\\lambda\\left(n\\right)\\right)^{2}'
    assert latex(divisor_sigma(x)) == '\\sigma\\left(x\\right)'
    assert latex(divisor_sigma(x) ** 2) == '\\sigma^{2}\\left(x\\right)'
    assert latex(divisor_sigma(x, y)) == '\\sigma_y\\left(x\\right)'
    assert latex(divisor_sigma(x, y) ** 2) == '\\sigma^{2}_y\\left(x\\right)'
    assert latex(udivisor_sigma(x)) == '\\sigma^*\\left(x\\right)'
    assert latex(udivisor_sigma(x) ** 2) == '\\sigma^*^{2}\\left(x\\right)'
    assert latex(udivisor_sigma(x, y)) == '\\sigma^*_y\\left(x\\right)'
    assert latex(udivisor_sigma(x, y) ** 2) == '\\sigma^*^{2}_y\\left(x\\right)'
    assert latex(primenu(n)) == '\\nu\\left(n\\right)'
    assert latex(primenu(n) ** 2) == '\\left(\\nu\\left(n\\right)\\right)^{2}'
    assert latex(primeomega(n)) == '\\Omega\\left(n\\right)'
    assert latex(primeomega(n) ** 2) == '\\left(\\Omega\\left(n\\right)\\right)^{2}'
    assert latex(LambertW(n)) == 'W\\left(n\\right)'
    assert latex(LambertW(n, -1)) == 'W_{-1}\\left(n\\right)'
    assert latex(LambertW(n, k)) == 'W_{k}\\left(n\\right)'
    assert latex(LambertW(n) * LambertW(n)) == 'W^{2}\\left(n\\right)'
    assert latex(Pow(LambertW(n), 2)) == 'W^{2}\\left(n\\right)'
    assert latex(LambertW(n) ** k) == 'W^{k}\\left(n\\right)'
    assert latex(LambertW(n, k) ** p) == 'W^{p}_{k}\\left(n\\right)'
    assert latex(Mod(x, 7)) == 'x \\bmod 7'
    assert latex(Mod(x + 1, 7)) == '\\left(x + 1\\right) \\bmod 7'
    assert latex(Mod(7, x + 1)) == '7 \\bmod \\left(x + 1\\right)'
    assert latex(Mod(2 * x, 7)) == '2 x \\bmod 7'
    assert latex(Mod(7, 2 * x)) == '7 \\bmod 2 x'
    assert latex(Mod(x, 7) + 1) == '\\left(x \\bmod 7\\right) + 1'
    assert latex(2 * Mod(x, 7)) == '2 \\left(x \\bmod 7\\right)'
    assert latex(Mod(7, 2 * x) ** n) == '\\left(7 \\bmod 2 x\\right)^{n}'
    fjlkd = Function('fjlkd')
    assert latex(fjlkd(x)) == '\\operatorname{fjlkd}{\\left(x \\right)}'
    assert latex(fjlkd) == '\\operatorname{fjlkd}'