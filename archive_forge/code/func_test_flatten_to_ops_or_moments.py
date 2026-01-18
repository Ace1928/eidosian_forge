from typing import cast
import pytest
import cirq
def test_flatten_to_ops_or_moments():
    operations = [cirq.GateOperation(cirq.testing.SingleQubitGate(), [cirq.NamedQubit(str(i))]) for i in range(10)]
    op_tree = [operations[0], cirq.Moment(operations[1:5]), operations[5:]]
    output = [operations[0], cirq.Moment(operations[1:5])] + operations[5:]
    assert list(cirq.flatten_to_ops_or_moments(op_tree)) == output
    assert list(cirq.flatten_op_tree(op_tree, preserve_moments=True)) == output
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_to_ops_or_moments(None))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_to_ops_or_moments(5))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_to_ops_or_moments([operations[0], (4,)]))