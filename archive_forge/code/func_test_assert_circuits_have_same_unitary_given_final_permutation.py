import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_circuits_have_same_unitary_given_final_permutation():
    q = cirq.LineQubit.range(5)
    expected = cirq.Circuit([cirq.Moment(cirq.CNOT(q[2], q[1]), cirq.CNOT(q[3], q[0]))])
    actual = cirq.Circuit([cirq.Moment(cirq.CNOT(q[2], q[1])), cirq.Moment(cirq.SWAP(q[0], q[2])), cirq.Moment(cirq.SWAP(q[0], q[1])), cirq.Moment(cirq.CNOT(q[3], q[2]))])
    qubit_map = {q[0]: q[2], q[2]: q[1], q[1]: q[0]}
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(actual, expected, qubit_map)
    qubit_map.update({q[2]: q[3]})
    with pytest.raises(ValueError, match="'qubit_map' must have the same set"):
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(actual, expected, qubit_map=qubit_map)
    bad_qubit_map = {q[0]: q[2], q[2]: q[4], q[4]: q[0]}
    with pytest.raises(ValueError, match="'qubit_map' must be a mapping"):
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(actual, expected, qubit_map=bad_qubit_map)