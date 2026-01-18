import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_kraus():
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    a, b = cirq.LineQubit.range(2)
    m = cirq.Moment()
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 1
    assert np.allclose(k[0], np.array([[1.0]]))
    m = cirq.Moment(cirq.S(a))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 1
    assert np.allclose(k[0], np.diag([1, 1j]))
    m = cirq.Moment(cirq.CNOT(a, b))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 1
    assert np.allclose(k[0], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    p = 0.1
    m = cirq.Moment(cirq.depolarize(p).on(a))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 4
    assert np.allclose(k[0], np.sqrt(1 - p) * I)
    assert np.allclose(k[1], np.sqrt(p / 3) * X)
    assert np.allclose(k[2], np.sqrt(p / 3) * Y)
    assert np.allclose(k[3], np.sqrt(p / 3) * Z)
    p = 0.2
    q = 0.3
    m = cirq.Moment(cirq.bit_flip(p).on(a), cirq.phase_flip(q).on(b))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 4
    assert np.allclose(k[0], np.sqrt((1 - p) * (1 - q)) * np.kron(I, I))
    assert np.allclose(k[1], np.sqrt(q * (1 - p)) * np.kron(I, Z))
    assert np.allclose(k[2], np.sqrt(p * (1 - q)) * np.kron(X, I))
    assert np.allclose(k[3], np.sqrt(p * q) * np.kron(X, Z))