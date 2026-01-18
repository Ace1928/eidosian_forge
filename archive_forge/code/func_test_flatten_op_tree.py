from typing import cast
import pytest
import cirq
def test_flatten_op_tree():
    operations = [cirq.GateOperation(cirq.testing.SingleQubitGate(), [cirq.NamedQubit(str(i))]) for i in range(10)]
    assert list(cirq.flatten_op_tree([[[]]])) == []
    assert list(cirq.flatten_op_tree(operations[0])) == operations[:1]
    assert list(cirq.flatten_op_tree(operations)) == operations
    assert list(cirq.flatten_op_tree((operations[0], operations[1:5], operations[5:]))) == operations
    assert list(cirq.flatten_op_tree((operations[0], cirq.Moment(operations[1:5]), operations[5:]))) == operations
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree(None))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree(5))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree([operations[0], (4,)]))