from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_gate_shape_protocol():
    """This test is only needed while the `_num_qubits_` and `_qid_shape_`
    methods are implemented as alternatives.  This can be removed once the
    deprecated `num_qubits` method is removed."""

    class NotImplementedGate1(cirq.Gate):

        def _num_qubits_(self):
            return NotImplemented

        def _qid_shape_(self):
            return NotImplemented

    class NotImplementedGate2(cirq.Gate):

        def _num_qubits_(self):
            return NotImplemented

    class NotImplementedGate3(cirq.Gate):

        def _qid_shape_(self):
            return NotImplemented

    class ShapeGate(cirq.Gate):

        def _num_qubits_(self):
            return NotImplemented

        def _qid_shape_(self):
            return (1, 2, 3)

    class QubitGate(cirq.Gate):

        def _num_qubits_(self):
            return 2

        def _qid_shape_(self):
            return NotImplemented
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.qid_shape(NotImplementedGate1())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.num_qubits(NotImplementedGate1())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = NotImplementedGate1().num_qubits()
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.qid_shape(NotImplementedGate2())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.num_qubits(NotImplementedGate2())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = NotImplementedGate2().num_qubits()
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.qid_shape(NotImplementedGate3())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        cirq.num_qubits(NotImplementedGate3())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = NotImplementedGate3().num_qubits()
    assert cirq.qid_shape(ShapeGate()) == (1, 2, 3)
    assert cirq.num_qubits(ShapeGate()) == 3
    assert ShapeGate().num_qubits() == 3
    assert cirq.qid_shape(QubitGate()) == (2, 2)
    assert cirq.num_qubits(QubitGate()) == 2
    assert QubitGate().num_qubits() == 2