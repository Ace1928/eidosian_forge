import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_once_with_qubits():
    qs = cirq.LineQubit.range(3)
    with pytest.raises(TypeError, match='no _decompose_with_context_ or _decompose_ method'):
        _ = cirq.decompose_once_with_qubits(NoMethod(), qs)
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once_with_qubits(DecomposeNotImplemented(), qs)
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once_with_qubits(DecomposeNone(), qs)
    assert cirq.decompose_once_with_qubits(NoMethod(), qs, 5) == 5
    assert cirq.decompose_once_with_qubits(DecomposeNotImplemented(), qs, None) is None
    assert cirq.decompose_once_with_qubits(NoMethod(), qs, NotImplemented) is NotImplemented
    assert cirq.decompose_once_with_qubits(DecomposeWithQubitsGiven(cirq.X.on_each), qs) == [cirq.X(cirq.LineQubit(0)), cirq.X(cirq.LineQubit(1)), cirq.X(cirq.LineQubit(2))]
    assert cirq.decompose_once_with_qubits(DecomposeWithQubitsGiven(lambda *qubits: cirq.Y(qubits[0])), qs) == [cirq.Y(cirq.LineQubit(0))]
    assert cirq.decompose_once_with_qubits(DecomposeWithQubitsGiven(lambda *qubits: (cirq.Y(q) for q in qubits)), qs) == [cirq.Y(cirq.LineQubit(0)), cirq.Y(cirq.LineQubit(1)), cirq.Y(cirq.LineQubit(2))]
    assert cirq.decompose_once_with_qubits(DecomposeQuditGate(), cirq.LineQid.for_qid_shape((1, 2, 3))) == [cirq.identity_each(*cirq.LineQid.for_qid_shape((1, 2, 3)))]

    def use_qubits_twice(*qubits):
        a = list(qubits)
        b = list(qubits)
        yield cirq.X.on_each(*a)
        yield cirq.Y.on_each(*b)
    assert cirq.decompose_once_with_qubits(DecomposeWithQubitsGiven(use_qubits_twice), (q for q in qs)) == list(cirq.X.on_each(*qs)) + list(cirq.Y.on_each(*qs))