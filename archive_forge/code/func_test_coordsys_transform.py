from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy
def test_coordsys_transform():
    p, q, r, s = symbols('p q r s')
    rel = {('first', 'second'): [(p, q), (q, -p)]}
    R2_pq = CoordSystem('first', R2_origin, [p, q], rel)
    R2_rs = CoordSystem('second', R2_origin, [r, s], rel)
    r, s = R2_rs.symbols
    assert R2_rs.transform(R2_pq) == Matrix([[-s], [r]])
    a, b = symbols('a b', positive=True)
    rel = {('first', 'second'): [(a,), (-a,)]}
    R2_a = CoordSystem('first', R2_origin, [a], rel)
    R2_b = CoordSystem('second', R2_origin, [b], rel)
    with raises(NotImplementedError):
        R2_b.transform(R2_a)
    c, d = symbols('c d')
    rel = {('first', 'second'): [(c,), (c ** 2,)]}
    R2_c = CoordSystem('first', R2_origin, [c], rel)
    R2_d = CoordSystem('second', R2_origin, [d], rel)
    with raises(ValueError):
        R2_d.transform(R2_c)
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2 * a, 3 * b)], ('C2', 'C3'): [(c, d), (3 * c, 2 * d)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c / 2, d / 3])
    assert C1.transform(C3) == Matrix([6 * a, 6 * b])
    assert C3.transform(C1) == Matrix([e / 6, f / 6])
    assert C3.transform(C2) == Matrix([e / 3, f / 2])
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2 * a, 3 * b + 1)], ('C3', 'C2'): [(e, f), (-e - 2, 2 * f)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c / 2, (d - 1) / 3])
    assert C1.transform(C3) == Matrix([-2 * a - 2, (3 * b + 1) / 2])
    assert C3.transform(C1) == Matrix([-e / 2 - 1, (2 * f - 1) / 3])
    assert C3.transform(C2) == Matrix([-e - 2, 2 * f])
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): Lambda((a, b), (2 * a, 3 * b + 1)), ('C3', 'C2'): Lambda((e, f), (-e - 2, 2 * f))}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c / 2, (d - 1) / 3])
    assert C1.transform(C3) == Matrix([-2 * a - 2, (3 * b + 1) / 2])
    assert C3.transform(C1) == Matrix([-e / 2 - 1, (2 * f - 1) / 3])
    assert C3.transform(C2) == Matrix([-e - 2, 2 * f])