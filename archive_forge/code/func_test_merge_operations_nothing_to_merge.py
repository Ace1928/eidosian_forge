from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_nothing_to_merge():

    def fail_if_called_func(*_):
        assert False
    c = cirq.Circuit()
    assert cirq.merge_operations(c, fail_if_called_func) == c
    q = cirq.LineQubit.range(3)
    c += cirq.Moment(cirq.CZ(*q[:2]))
    assert cirq.merge_operations(c, fail_if_called_func) == c
    c += cirq.Moment(cirq.X(q[2]), cirq.global_phase_operation(1j))
    assert cirq.merge_operations(c, fail_if_called_func) == c
    c += cirq.Moment(cirq.CNOT(*q[:2]).with_tags('ignore'))
    assert cirq.merge_operations(c, fail_if_called_func, tags_to_ignore=['ignore']) == c