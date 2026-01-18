from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def optimize_field_generator(z):
    p = z.min_polynomial()
    assert p.base_ring() == ZZ
    a, n = (p.leading_coefficient(), p.degree())
    x = PolynomialRing(QQ, 'x').gen()
    q = a ** (n - 1) * p(x / a)
    root_of_q = a * z(1000)
    K = NumberField(q, 'x')
    F, F_to_K, K_to_F = K.optimized_representation()
    w = F_to_K(F.gen()).polynomial()(root_of_q)
    f = F.defining_polynomial()
    f = f.denominator() * f
    return ExactAlgebraicNumber(f.change_ring(ZZ), w)