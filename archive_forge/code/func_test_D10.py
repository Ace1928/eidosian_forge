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
@XFAIL
def test_D10():
    raise NotImplementedError('translate D8 to C')