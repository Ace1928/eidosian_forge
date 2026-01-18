from sympy.assumptions.ask import Q, ask
from sympy.assumptions.refine import refine
from sympy.concrete.products import product
from sympy.core import EulerGamma
from sympy.core.evalf import N
from sympy.core.function import (Derivative, Function, Lambda, Subs,
from sympy.core.mul import Mul
from sympy.core.numbers import (AlgebraicNumber, E, I, Rational, igcd,
from sympy.core.relational import Eq, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.functions.combinatorial.factorials import (rf, binomial,
from sympy.functions.combinatorial.numbers import bernoulli, fibonacci
from sympy.functions.elementary.complexes import (conjugate, im, re,
from sympy.functions.elementary.exponential import LambertW, exp, log
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh,
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, asin,
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e,
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.polynomials import (assoc_legendre,
from sympy.functions.special.zeta_functions import polylog
from sympy.geometry.util import idiff
from sympy.logic.boolalg import And
from sympy.matrices.dense import hessian, wronskian
from sympy.matrices.expressions.matmul import MatMul
from sympy.ntheory.continued_fraction import (
from sympy.ntheory.factor_ import factorint, totient
from sympy.ntheory.generate import primerange
from sympy.ntheory.partitions_ import npartitions
from sympy.polys.domains.integerring import ZZ
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.partfrac import apart
from sympy.polys.polytools import Poly, factor, gcd, resultant
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import FiniteSet, Intersection, Interval, Union
from sympy.simplify.combsimp import combsimp
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.radsimp import radsimp
from sympy.simplify.simplify import logcombine, simplify
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve
import mpmath
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import Ci, Si, erf
from sympy.functions.special.zeta_functions import zeta
from sympy.testing.pytest import (XFAIL, slow, SKIP, skip, ON_CI,
from sympy.utilities.iterables import partitions
from mpmath import mpi, mpc
from sympy.matrices import Matrix, GramSchmidt, eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix, block_collapse
from sympy.matrices.expressions import MatrixSymbol, ZeroMatrix
from sympy.physics.quantum import Commutator
from sympy.polys.rings import PolyRing
from sympy.polys.fields import FracField
from sympy.polys.solvers import solve_lin_sys
from sympy.concrete import Sum
from sympy.concrete.products import Product
from sympy.integrals import integrate
from sympy.integrals.transforms import laplace_transform,\
from sympy.solvers.recurr import rsolve
from sympy.solvers.solveset import solveset, solveset_real, linsolve
from sympy.solvers.ode import dsolve
from sympy.core.relational import Equality
from itertools import islice, takewhile
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.calculus.util import minimum
def test_M38():
    a, b, c = symbols('a, b, c')
    domain = FracField([a, b, c], ZZ).to_domain()
    ring = PolyRing('k1:50', domain)
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27, k28, k29, k30, k31, k32, k33, k34, k35, k36, k37, k38, k39, k40, k41, k42, k43, k44, k45, k46, k47, k48, k49 = ring.gens
    system = [-b * k8 / a + c * k8 / a, -b * k11 / a + c * k11 / a, -b * k10 / a + c * k10 / a + k2, -k3 - b * k9 / a + c * k9 / a, -b * k14 / a + c * k14 / a, -b * k15 / a + c * k15 / a, -b * k18 / a + c * k18 / a - k2, -b * k17 / a + c * k17 / a, -b * k16 / a + c * k16 / a + k4, -b * k13 / a + c * k13 / a - b * k21 / a + c * k21 / a + b * k5 / a - c * k5 / a, b * k44 / a - c * k44 / a, -b * k45 / a + c * k45 / a, -b * k20 / a + c * k20 / a, -b * k44 / a + c * k44 / a, b * k46 / a - c * k46 / a, b ** 2 * k47 / a ** 2 - 2 * b * c * k47 / a ** 2 + c ** 2 * k47 / a ** 2, k3, -k4, -b * k12 / a + c * k12 / a - a * k6 / b + c * k6 / b, -b * k19 / a + c * k19 / a + a * k7 / c - b * k7 / c, b * k45 / a - c * k45 / a, -b * k46 / a + c * k46 / a, -k48 + c * k48 / a + c * k48 / b - c ** 2 * k48 / (a * b), -k49 + b * k49 / a + b * k49 / c - b ** 2 * k49 / (a * c), a * k1 / b - c * k1 / b, a * k4 / b - c * k4 / b, a * k3 / b - c * k3 / b + k9, -k10 + a * k2 / b - c * k2 / b, a * k7 / b - c * k7 / b, -k9, k11, b * k12 / a - c * k12 / a + a * k6 / b - c * k6 / b, a * k15 / b - c * k15 / b, k10 + a * k18 / b - c * k18 / b, -k11 + a * k17 / b - c * k17 / b, a * k16 / b - c * k16 / b, -a * k13 / b + c * k13 / b + a * k21 / b - c * k21 / b + a * k5 / b - c * k5 / b, -a * k44 / b + c * k44 / b, a * k45 / b - c * k45 / b, a * k14 / c - b * k14 / c + a * k20 / b - c * k20 / b, a * k44 / b - c * k44 / b, -a * k46 / b + c * k46 / b, -k47 + c * k47 / a + c * k47 / b - c ** 2 * k47 / (a * b), a * k19 / b - c * k19 / b, -a * k45 / b + c * k45 / b, a * k46 / b - c * k46 / b, a ** 2 * k48 / b ** 2 - 2 * a * c * k48 / b ** 2 + c ** 2 * k48 / b ** 2, -k49 + a * k49 / b + a * k49 / c - a ** 2 * k49 / (b * c), k16, -k17, -a * k1 / c + b * k1 / c, -k16 - a * k4 / c + b * k4 / c, -a * k3 / c + b * k3 / c, k18 - a * k2 / c + b * k2 / c, b * k19 / a - c * k19 / a - a * k7 / c + b * k7 / c, -a * k6 / c + b * k6 / c, -a * k8 / c + b * k8 / c, -a * k11 / c + b * k11 / c + k17, -a * k10 / c + b * k10 / c - k18, -a * k9 / c + b * k9 / c, -a * k14 / c + b * k14 / c - a * k20 / b + c * k20 / b, -a * k13 / c + b * k13 / c + a * k21 / c - b * k21 / c - a * k5 / c + b * k5 / c, a * k44 / c - b * k44 / c, -a * k45 / c + b * k45 / c, -a * k44 / c + b * k44 / c, a * k46 / c - b * k46 / c, -k47 + b * k47 / a + b * k47 / c - b ** 2 * k47 / (a * c), -a * k12 / c + b * k12 / c, a * k45 / c - b * k45 / c, -a * k46 / c + b * k46 / c, -k48 + a * k48 / b + a * k48 / c - a ** 2 * k48 / (b * c), a ** 2 * k49 / c ** 2 - 2 * a * b * k49 / c ** 2 + b ** 2 * k49 / c ** 2, k8, k11, -k15, k10 - k18, -k17, k9, -k16, -k29, k14 - k32, -k21 + k23 - k31, -k24 - k30, -k35, k44, -k45, k36, k13 - k23 + k39, -k20 + k38, k25 + k37, b * k26 / a - c * k26 / a - k34 + k42, -2 * k44, k45, k46, b * k47 / a - c * k47 / a, k41, k44, -k46, -b * k47 / a + c * k47 / a, k12 + k24, -k19 - k25, -a * k27 / b + c * k27 / b - k33, k45, -k46, -a * k48 / b + c * k48 / b, a * k28 / c - b * k28 / c + k40, -k45, k46, a * k48 / b - c * k48 / b, a * k49 / c - b * k49 / c, -a * k49 / c + b * k49 / c, -k1, -k4, -k3, k15, k18 - k2, k17, k16, k22, k25 - k7, k24 + k30, k21 + k23 - k31, k28, -k44, k45, -k30 - k6, k20 + k32, k27 + b * k33 / a - c * k33 / a, k44, -k46, -b * k47 / a + c * k47 / a, -k36, k31 - k39 - k5, -k32 - k38, k19 - k37, k26 - a * k34 / b + c * k34 / b - k42, k44, -2 * k45, k46, a * k48 / b - c * k48 / b, a * k35 / c - b * k35 / c - k41, -k44, k46, b * k47 / a - c * k47 / a, -a * k49 / c + b * k49 / c, -k40, k45, -k46, -a * k48 / b + c * k48 / b, a * k49 / c - b * k49 / c, k1, k4, k3, -k8, -k11, -k10 + k2, -k9, k37 + k7, -k14 - k38, -k22, -k25 - k37, -k24 + k6, -k13 - k23 + k39, -k28 + b * k40 / a - c * k40 / a, k44, -k45, -k27, -k44, k46, b * k47 / a - c * k47 / a, k29, k32 + k38, k31 - k39 + k5, -k12 + k30, k35 - a * k41 / b + c * k41 / b, -k44, k45, -k26 + k34 + a * k42 / c - b * k42 / c, k44, k45, -2 * k46, -b * k47 / a + c * k47 / a, -a * k48 / b + c * k48 / b, a * k49 / c - b * k49 / c, k33, -k45, k46, a * k48 / b - c * k48 / b, -a * k49 / c + b * k49 / c]
    solution = {k49: 0, k48: 0, k47: 0, k46: 0, k45: 0, k44: 0, k41: 0, k40: 0, k38: 0, k37: 0, k36: 0, k35: 0, k33: 0, k32: 0, k30: 0, k29: 0, k28: 0, k27: 0, k25: 0, k24: 0, k22: 0, k21: 0, k20: 0, k19: 0, k18: 0, k17: 0, k16: 0, k15: 0, k14: 0, k13: 0, k12: 0, k11: 0, k10: 0, k9: 0, k8: 0, k7: 0, k6: 0, k5: 0, k4: 0, k3: 0, k2: 0, k1: 0, k34: b / c * k42, k31: k39, k26: a / c * k42, k23: k39}
    assert solve_lin_sys(system, ring) == solution