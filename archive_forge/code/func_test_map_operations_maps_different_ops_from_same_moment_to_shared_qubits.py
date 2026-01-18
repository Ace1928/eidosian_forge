from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_operations_maps_different_ops_from_same_moment_to_shared_qubits():
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.H.on_each(q[:2]))
    c_mapped = cirq.map_operations(c, lambda op, _: op.controlled_by(q[2]), raise_if_add_qubits=False)
    cirq.testing.assert_same_circuits(c_mapped, cirq.Circuit(cirq.H(q[0]).controlled_by(q[2]), cirq.H(q[1]).controlled_by(q[2])))