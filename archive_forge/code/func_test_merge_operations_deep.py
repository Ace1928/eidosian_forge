from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_deep():
    q = cirq.LineQubit.range(2)
    h_cz_y = [cirq.H(q[0]), cirq.CZ(*q), cirq.Y(q[1])]
    m_cz_m = [cirq.Moment(), cirq.Moment(cirq.CZ(*q)), cirq.Moment()]
    c_orig = cirq.Circuit(h_cz_y, cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.Y(q[1])), cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags('ignore'), [cirq.CNOT(*q), cirq.CNOT(*q)], cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(4), [cirq.CNOT(*q), cirq.CZ(*q), cirq.CNOT(*q)], cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(5).with_tags('preserve_tag'))
    c_expected = cirq.Circuit(m_cz_m, cirq.Moment(cirq.X(q[0]).with_tags('ignore')), cirq.CircuitOperation(cirq.FrozenCircuit(h_cz_y)).repeat(6).with_tags('ignore'), [cirq.CNOT(*q), cirq.CNOT(*q)], cirq.CircuitOperation(cirq.FrozenCircuit(m_cz_m)).repeat(4), [cirq.CZ(*q), cirq.Moment(), cirq.Moment()], cirq.CircuitOperation(cirq.FrozenCircuit(m_cz_m)).repeat(5).with_tags('preserve_tag'), strategy=cirq.InsertStrategy.NEW)

    def merge_func(op1, op2):
        """Artificial example where a CZ will absorb any merge-able operation."""
        for op in [op1, op2]:
            if op.gate == cirq.CZ:
                return op
        return None
    cirq.testing.assert_same_circuits(cirq.merge_operations(c_orig, merge_func, tags_to_ignore=['ignore'], deep=True), c_expected)