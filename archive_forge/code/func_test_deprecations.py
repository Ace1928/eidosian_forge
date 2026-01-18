from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy
def test_deprecations():
    m = Manifold('M', 2)
    p = Patch('P', m)
    with warns_deprecated_sympy():
        CoordSystem('Car2d', p, names=['x', 'y'])
    with warns_deprecated_sympy():
        c = CoordSystem('Car2d', p, ['x', 'y'])
    with warns_deprecated_sympy():
        list(m.patches)
    with warns_deprecated_sympy():
        list(c.transforms)