from typing import List
import numpy as np
import pytest
import cirq
def test_merge_k_qubit_unitaries_deep_recurses_on_large_circuit_op():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q[0]), cirq.H(q[0]), cirq.CNOT(*q))))
    c_expected = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q[0]), cirq.H(q[0]))).with_tags('merged'), cirq.CNOT(*q))))
    c_new = cirq.merge_k_qubit_unitaries(c_orig, context=cirq.TransformerContext(deep=True), k=1, rewriter=lambda op: op.with_tags('merged'))
    cirq.testing.assert_same_circuits(c_new, c_expected)