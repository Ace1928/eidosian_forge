import pytest
import cirq
def test_neighborhood():
    assert cirq.LineQubit(1).neighbors() == {cirq.LineQubit(0), cirq.LineQubit(2)}
    restricted_qubits = [cirq.LineQubit(2), cirq.LineQubit(3)]
    assert cirq.LineQubit(1).neighbors(restricted_qubits) == {cirq.LineQubit(2)}