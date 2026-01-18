from sympy.matrices.dense import eye, Matrix
from sympy.tensor.tensor import tensor_indices, TensorHead, tensor_heads, \
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, \
from sympy import Symbol
def execute_gamma_simplify_tests_for_function(tfunc, D):
    """
    Perform tests to check if sfunc is able to simplify gamma matrix expressions.

    Parameters
    ==========

    `sfunc`     a function to simplify a `TIDS`, shall return the simplified `TIDS`.
    `D`         the number of dimension (in most cases `D=4`).

    """
    mu, nu, rho, sigma = tensor_indices('mu, nu, rho, sigma', LorentzIndex)
    a1, a2, a3, a4, a5, a6 = tensor_indices('a1:7', LorentzIndex)
    mu11, mu12, mu21, mu31, mu32, mu41, mu51, mu52 = tensor_indices('mu11, mu12, mu21, mu31, mu32, mu41, mu51, mu52', LorentzIndex)
    mu61, mu71, mu72 = tensor_indices('mu61, mu71, mu72', LorentzIndex)
    m0, m1, m2, m3, m4, m5, m6 = tensor_indices('m0:7', LorentzIndex)

    def g(xx, yy):
        return (G(xx) * G(yy) + G(yy) * G(xx)) / 2
    if D == 4:
        t = G(a1) * G(mu11) * G(a2) * G(mu21) * G(-a1) * G(mu31) * G(-a2)
        assert _is_tensor_eq(tfunc(t), -4 * G(mu11) * G(mu31) * G(mu21) - 4 * G(mu31) * G(mu11) * G(mu21))
        t = G(a1) * G(mu11) * G(mu12) * G(a2) * G(mu21) * G(a3) * G(mu31) * G(mu32) * G(a4) * G(mu41) * G(-a2) * G(mu51) * G(mu52) * G(-a1) * G(mu61) * G(-a3) * G(mu71) * G(mu72) * G(-a4)
        assert _is_tensor_eq(tfunc(t), 16 * G(mu31) * G(mu32) * G(mu72) * G(mu71) * G(mu11) * G(mu52) * G(mu51) * G(mu12) * G(mu61) * G(mu21) * G(mu41) + 16 * G(mu31) * G(mu32) * G(mu72) * G(mu71) * G(mu12) * G(mu51) * G(mu52) * G(mu11) * G(mu61) * G(mu21) * G(mu41) + 16 * G(mu71) * G(mu72) * G(mu32) * G(mu31) * G(mu11) * G(mu52) * G(mu51) * G(mu12) * G(mu61) * G(mu21) * G(mu41) + 16 * G(mu71) * G(mu72) * G(mu32) * G(mu31) * G(mu12) * G(mu51) * G(mu52) * G(mu11) * G(mu61) * G(mu21) * G(mu41))

    def add_delta(ne):
        return ne * eye(4)
    t = G(mu) * G(-mu)
    ts = add_delta(D)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(mu) * G(nu) * G(-mu) * G(-nu)
    ts = add_delta(2 * D - D ** 2)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(mu) * G(nu) * G(-nu) * G(-mu)
    ts = add_delta(D ** 2)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(mu) * G(nu) * G(-rho) * G(-nu) * G(-mu) * G(rho)
    ts = add_delta(4 * D - 4 * D ** 2 + D ** 3)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(mu) * G(nu) * G(rho) * G(-rho) * G(-nu) * G(-mu)
    ts = add_delta(D ** 3)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(a1) * G(a2) * G(a3) * G(a4) * G(-a3) * G(-a1) * G(-a2) * G(-a4)
    ts = add_delta(-8 * D + 16 * D ** 2 - 8 * D ** 3 + D ** 4)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(-mu) * G(-nu) * G(-rho) * G(-sigma) * G(nu) * G(mu) * G(sigma) * G(rho)
    ts = add_delta(-16 * D + 24 * D ** 2 - 8 * D ** 3 + D ** 4)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(-mu) * G(nu) * G(-rho) * G(sigma) * G(rho) * G(-nu) * G(mu) * G(-sigma)
    ts = add_delta(8 * D - 12 * D ** 2 + 6 * D ** 3 - D ** 4)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(a1) * G(a2) * G(a3) * G(a4) * G(a5) * G(-a3) * G(-a2) * G(-a1) * G(-a5) * G(-a4)
    ts = add_delta(64 * D - 112 * D ** 2 + 60 * D ** 3 - 12 * D ** 4 + D ** 5)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(a1) * G(a2) * G(a3) * G(a4) * G(a5) * G(-a3) * G(-a1) * G(-a2) * G(-a4) * G(-a5)
    ts = add_delta(64 * D - 120 * D ** 2 + 72 * D ** 3 - 16 * D ** 4 + D ** 5)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(a1) * G(a2) * G(a3) * G(a4) * G(a5) * G(a6) * G(-a3) * G(-a2) * G(-a1) * G(-a6) * G(-a5) * G(-a4)
    ts = add_delta(416 * D - 816 * D ** 2 + 528 * D ** 3 - 144 * D ** 4 + 18 * D ** 5 - D ** 6)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(a1) * G(a2) * G(a3) * G(a4) * G(a5) * G(a6) * G(-a2) * G(-a3) * G(-a1) * G(-a6) * G(-a4) * G(-a5)
    ts = add_delta(416 * D - 848 * D ** 2 + 584 * D ** 3 - 172 * D ** 4 + 22 * D ** 5 - D ** 6)
    assert _is_tensor_eq(tfunc(t), ts)
    t = G(mu) * G(nu) * G(rho) * G(sigma) * G(-mu)
    assert _is_tensor_eq(tfunc(t), -2 * G(sigma) * G(rho) * G(nu) + (4 - D) * G(nu) * G(rho) * G(sigma))
    t = G(mu) * G(nu) * G(-mu)
    assert _is_tensor_eq(tfunc(t), (2 - D) * G(nu))
    t = G(mu) * G(nu) * G(rho) * G(-mu)
    assert _is_tensor_eq(tfunc(t), 2 * G(nu) * G(rho) + 2 * G(rho) * G(nu) - (4 - D) * G(nu) * G(rho))
    t = 2 * G(m2) * G(m0) * G(m1) * G(-m0) * G(-m1)
    st = tfunc(t)
    assert _is_tensor_eq(st, D * (-2 * D + 4) * G(m2))
    t = G(m2) * G(m0) * G(m1) * G(-m0) * G(-m2)
    st = tfunc(t)
    assert _is_tensor_eq(st, (-D + 2) ** 2 * G(m1))
    t = G(m0) * G(m1) * G(m2) * G(m3) * G(-m1)
    st = tfunc(t)
    assert _is_tensor_eq(st, (D - 4) * G(m0) * G(m2) * G(m3) + 4 * G(m0) * g(m2, m3))
    t = G(m0) * G(m1) * G(m2) * G(m3) * G(-m1) * G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, (D - 4) ** 2 * G(m2) * G(m3) + (8 * D - 16) * g(m2, m3))
    t = G(m2) * G(m0) * G(m1) * G(-m2) * G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, ((-D + 2) * (D - 4) + 4) * G(m1))
    t = G(m3) * G(m1) * G(m0) * G(m2) * G(-m3) * G(-m0) * G(-m2)
    st = tfunc(t)
    assert _is_tensor_eq(st, (-4 * D + (-D + 2) ** 2 * (D - 4) + 8) * G(m1))
    t = 2 * G(m0) * G(m1) * G(m2) * G(m3) * G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, (-2 * D + 8) * G(m1) * G(m2) * G(m3) - 4 * G(m3) * G(m2) * G(m1))
    t = G(m5) * G(m0) * G(m1) * G(m4) * G(m2) * G(-m4) * G(m3) * G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, (-D + 2) * (-D + 4) * G(m5) * G(m1) * G(m2) * G(m3) + (2 * D - 4) * G(m5) * G(m3) * G(m2) * G(m1))
    t = -G(m0) * G(m1) * G(m2) * G(m3) * G(-m0) * G(m4)
    st = tfunc(t)
    assert _is_tensor_eq(st, (D - 4) * G(m1) * G(m2) * G(m3) * G(m4) + 2 * G(m3) * G(m2) * G(m1) * G(m4))
    t = G(-m5) * G(m0) * G(m1) * G(m2) * G(m3) * G(m4) * G(-m0) * G(m5)
    st = tfunc(t)
    result1 = ((-D + 4) ** 2 + 4) * G(m1) * G(m2) * G(m3) * G(m4) + (4 * D - 16) * G(m3) * G(m2) * G(m1) * G(m4) + (4 * D - 16) * G(m4) * G(m1) * G(m2) * G(m3) + 4 * G(m2) * G(m1) * G(m4) * G(m3) + 4 * G(m3) * G(m4) * G(m1) * G(m2) + 4 * G(m4) * G(m3) * G(m2) * G(m1)
    result2 = 8 * G(m1) * G(m2) * G(m3) * G(m4) + 8 * G(m4) * G(m3) * G(m2) * G(m1)
    if D == 4:
        assert _is_tensor_eq(st, result1) or _is_tensor_eq(st, result2)
    else:
        assert _is_tensor_eq(st, result1)
    t = G(m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, t)
    t = -7 * G(m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, t)
    t = 224 * G(m0) * G(m1) * G(-m2) * G(m3)
    st = tfunc(t)
    assert _is_tensor_eq(st, t)