from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)
from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn
from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R
def test_linear_system_symbols_doesnt_hang_2():
    M = Matrix([[66, 24, 39, 50, 88, 40, 37, 96, 16, 65, 31, 11, 37, 72, 16, 19, 55, 37, 28, 76], [10, 93, 34, 98, 59, 44, 67, 74, 74, 94, 71, 61, 60, 23, 6, 2, 57, 8, 29, 78], [19, 91, 57, 13, 64, 65, 24, 53, 77, 34, 85, 58, 87, 39, 39, 7, 36, 67, 91, 3], [74, 70, 15, 53, 68, 43, 86, 83, 81, 72, 25, 46, 67, 17, 59, 25, 78, 39, 63, 6], [69, 40, 67, 21, 67, 40, 17, 13, 93, 44, 46, 89, 62, 31, 30, 38, 18, 20, 12, 81], [50, 22, 74, 76, 34, 45, 19, 76, 28, 28, 11, 99, 97, 82, 8, 46, 99, 57, 68, 35], [58, 18, 45, 88, 10, 64, 9, 34, 90, 82, 17, 41, 43, 81, 45, 83, 22, 88, 24, 39], [42, 21, 70, 68, 6, 33, 64, 81, 83, 15, 86, 75, 86, 17, 77, 34, 62, 72, 20, 24], [7, 8, 2, 72, 71, 52, 96, 5, 32, 51, 31, 36, 79, 88, 25, 77, 29, 26, 33, 13], [19, 31, 30, 85, 81, 39, 63, 28, 19, 12, 16, 49, 37, 66, 38, 13, 3, 71, 61, 51], [29, 82, 80, 49, 26, 85, 1, 37, 2, 74, 54, 82, 26, 47, 54, 9, 35, 0, 99, 40], [15, 49, 82, 91, 93, 57, 45, 25, 45, 97, 15, 98, 48, 52, 66, 24, 62, 54, 97, 37], [62, 23, 73, 53, 52, 86, 28, 38, 0, 74, 92, 38, 97, 70, 71, 29, 26, 90, 67, 45], [2, 32, 23, 24, 71, 37, 25, 71, 5, 41, 97, 65, 93, 13, 65, 45, 25, 88, 69, 50], [40, 56, 1, 29, 79, 98, 79, 62, 37, 28, 45, 47, 3, 1, 32, 74, 98, 35, 84, 32], [33, 15, 87, 79, 65, 9, 14, 63, 24, 19, 46, 28, 74, 20, 29, 96, 84, 91, 93, 1], [97, 18, 12, 52, 1, 2, 50, 14, 52, 76, 19, 82, 41, 73, 51, 79, 13, 3, 82, 96], [40, 28, 52, 10, 10, 71, 56, 78, 82, 5, 29, 48, 1, 26, 16, 18, 50, 76, 86, 52], [38, 89, 83, 43, 29, 52, 90, 77, 57, 0, 67, 20, 81, 88, 48, 96, 88, 58, 14, 3]])
    syms = x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18 = symbols('x:19')
    sol = {x0: -S(1967374186044955317099186851240896179) / 3166636564687820453598895768302256588, x1: -S(84268280268757263347292368432053826) / 791659141171955113399723942075564147, x2: -S(229962957341664730974463872411844965) / 1583318282343910226799447884151128294, x3: S(990156781744251750886760432229180537) / 6333273129375640907197791536604513176, x4: -S(2169830351210066092046760299593096265) / 18999819388126922721593374609813539528, x5: S(4680868883477577389628494526618745355) / 9499909694063461360796687304906769764, x6: -S(1590820774344371990683178396480879213) / 3166636564687820453598895768302256588, x7: -S(54104723404825537735226491634383072) / 339282489073695048599881689460956063, x8: S(3182076494196560075964847771774733847) / 6333273129375640907197791536604513176, x9: -S(10870817431029210431989147852497539675) / 18999819388126922721593374609813539528, x10: -S(13118019242576506476316318268573312603) / 18999819388126922721593374609813539528, x11: -S(5173852969886775824855781403820641259) / 4749954847031730680398343652453384882, x12: S(4261112042731942783763341580651820563) / 4749954847031730680398343652453384882, x13: -S(821833082694661608993818117038209051) / 6333273129375640907197791536604513176, x14: S(906881575107250690508618713632090559) / 904753304196520129599684505229216168, x15: -S(732162528717458388995329317371283987) / 6333273129375640907197791536604513176, x16: S(4524215476705983545537087360959896817) / 9499909694063461360796687304906769764, x17: -S(3898571347562055611881270844646055217) / 6333273129375640907197791536604513176, x18: S(7513502486176995632751685137907442269) / 18999819388126922721593374609813539528}
    eqs = list(M * Matrix(syms + (1,)))
    assert solve(eqs, syms) == sol
    y = Symbol('y')
    eqs = list(y * M * Matrix(syms + (1,)))
    assert solve(eqs, syms) == sol