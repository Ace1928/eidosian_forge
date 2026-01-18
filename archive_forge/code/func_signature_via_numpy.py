import spherogram
import snappy
import numpy as np
import mpmath
from sage.all import PolynomialRing, LaurentPolynomialRing, RR, ZZ, RealField, ComplexField, matrix, arccos, exp
def signature_via_numpy(A):
    CC = ComplexField(53)
    A = A.change_ring(CC).numpy()
    assert np.linalg.norm(A - A.conjugate().transpose()) < 1e-09
    eigs = np.linalg.eigh(A)[0]
    smallest = min(np.abs(eigs))
    assert smallest > 1e-05
    return np.sum(eigs > 0) - np.sum(eigs < 0)