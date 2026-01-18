from __future__ import division
import pytest
from mpmath import *
def hilbert_cmplx(n):
    A = hilbert(2 * n, n)
    v = randmatrix(2 * n, 2, min=-1, max=1)
    v = v.apply(lambda x: exp(1j * pi() * x))
    A = diag(v[:, 0]) * A * diag(v[:n, 1])
    return A