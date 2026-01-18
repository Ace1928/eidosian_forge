import numpy as np
import pytest
import sympy
import cirq
def test_commutes_on_matrices():
    I, X, Y, Z = (cirq.unitary(A) for A in (cirq.I, cirq.X, cirq.Y, cirq.Z))
    IX, IY = (np.kron(I, A) for A in (X, Y))
    XI, YI, ZI = (np.kron(A, I) for A in (X, Y, Z))
    XX, YY, ZZ = (np.kron(A, A) for A in (X, Y, Z))
    for A in (X, Y, Z):
        assert cirq.commutes(I, A)
        assert cirq.commutes(A, A)
        assert cirq.commutes(I, XX, default='default') == 'default'
    for A, B in [(X, Y), (X, Z), (Z, Y), (IX, IY), (XI, ZI)]:
        assert not cirq.commutes(A, B)
        assert not cirq.commutes(A, B, atol=1)
        assert cirq.commutes(A, B, atol=2)
    for A, B in [(XX, YY), (XX, ZZ), (ZZ, YY), (IX, YI), (IX, IX), (ZI, IY)]:
        assert cirq.commutes(A, B)