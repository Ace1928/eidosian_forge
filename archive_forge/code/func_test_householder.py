from __future__ import division
import pytest
from mpmath import *
def test_householder():
    mp.dps = 15
    A, b = (A8, b8)
    H, p, x, r = householder(extend(A, b))
    assert H == matrix([[mpf('3.0'), mpf('-2.0'), mpf('-1.0'), 0], [-1.0, mpf('3.333333333333333'), mpf('-2.9999999999999991'), mpf('2.0')], [-1.0, mpf('-0.66666666666666674'), mpf('2.8142135623730948'), mpf('-2.8284271247461898')], [1.0, mpf('-1.3333333333333333'), mpf('-0.20000000000000018'), mpf('4.2426406871192857')]])
    assert p == [-2, -2, mpf('-1.4142135623730949')]
    assert round(norm(r, 2), 10) == 4.2426406871
    y = [102.102, 58.344, 36.463, 24.31, 17.017, 12.376, 9.282, 7.14, 5.61, 4.488, 3.6465, 3.003]

    def coeff(n):
        A = []
        for i in range(1, 13):
            A.append([1.0 / (i + j - 1) for j in range(1, n + 1)])
        return matrix(A)
    residuals = []
    refres = []
    for n in range(2, 7):
        A = coeff(n)
        H, p, x, r = householder(extend(A, y))
        x = matrix(x)
        y = matrix(y)
        residuals.append(norm(r, 2))
        refres.append(norm(residual(A, x, y), 2))
    assert [round(res, 10) for res in residuals] == [15.1733888877, 0.8237807321, 0.302645887, 0.0260109244, 0.00058654]
    assert norm(matrix(residuals) - matrix(refres), inf) < 1e-13

    def hilbert_cmplx(n):
        A = hilbert(2 * n, n)
        v = randmatrix(2 * n, 2, min=-1, max=1)
        v = v.apply(lambda x: exp(1j * pi() * x))
        A = diag(v[:, 0]) * A * diag(v[:n, 1])
        return A
    residuals_cmplx = []
    refres_cmplx = []
    for n in range(2, 10):
        A = hilbert_cmplx(n)
        H, p, x, r = householder(A.copy())
        residuals_cmplx.append(norm(r, 2))
        refres_cmplx.append(norm(residual(A[:, :n - 1], x, A[:, n - 1]), 2))
    assert norm(matrix(residuals_cmplx) - matrix(refres_cmplx), inf) < 1e-13