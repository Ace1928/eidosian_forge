from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy
@nocache_fail
def test_covar_deriv():
    ch = metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    assert cvd(R2.x) == 1
    assert cvd(R2.x * R2.e_x) == R2.e_x
    cvd = CovarDerivativeOp(R2.x * R2.e_x, ch)
    assert cvd(R2.x) == R2.x
    assert cvd(R2.x * R2.e_x) == R2.x * R2.e_x