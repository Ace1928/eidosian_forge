from sympy.core import Lambda, Symbol, symbols
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy
def test_helpers_and_coordinate_dependent():
    one_form = R2.dr + R2.dx
    two_form = Differential(R2.x * R2.dr + R2.r * R2.dx)
    three_form = Differential(R2.y * two_form) + Differential(R2.x * Differential(R2.r * R2.dr))
    metric = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dy, R2.dy)
    metric_ambig = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dr, R2.dr)
    misform_a = TensorProduct(R2.dr, R2.dr) + R2.dr
    misform_b = R2.dr ** 4
    misform_c = R2.dx * R2.dy
    twoform_not_sym = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dx, R2.dy)
    twoform_not_TP = WedgeProduct(R2.dx, R2.dy)
    one_vector = R2.e_x + R2.e_y
    two_vector = TensorProduct(R2.e_x, R2.e_y)
    three_vector = TensorProduct(R2.e_x, R2.e_y, R2.e_x)
    two_wp = WedgeProduct(R2.e_x, R2.e_y)
    assert covariant_order(one_form) == 1
    assert covariant_order(two_form) == 2
    assert covariant_order(three_form) == 3
    assert covariant_order(two_form + metric) == 2
    assert covariant_order(two_form + metric_ambig) == 2
    assert covariant_order(two_form + twoform_not_sym) == 2
    assert covariant_order(two_form + twoform_not_TP) == 2
    assert contravariant_order(one_vector) == 1
    assert contravariant_order(two_vector) == 2
    assert contravariant_order(three_vector) == 3
    assert contravariant_order(two_vector + two_wp) == 2
    raises(ValueError, lambda: covariant_order(misform_a))
    raises(ValueError, lambda: covariant_order(misform_b))
    raises(ValueError, lambda: covariant_order(misform_c))
    assert twoform_to_matrix(metric) == Matrix([[1, 0], [0, 1]])
    assert twoform_to_matrix(twoform_not_sym) == Matrix([[1, 0], [1, 0]])
    assert twoform_to_matrix(twoform_not_TP) == Matrix([[0, -1], [1, 0]])
    raises(ValueError, lambda: twoform_to_matrix(one_form))
    raises(ValueError, lambda: twoform_to_matrix(three_form))
    raises(ValueError, lambda: twoform_to_matrix(metric_ambig))
    raises(ValueError, lambda: metric_to_Christoffel_1st(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Christoffel_2nd(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Riemann_components(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Ricci_components(twoform_not_sym))