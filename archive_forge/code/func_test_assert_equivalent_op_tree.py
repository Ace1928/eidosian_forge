import pytest
import cirq
from cirq.testing import assert_equivalent_op_tree
def test_assert_equivalent_op_tree():
    assert_equivalent_op_tree([], [])
    a = cirq.NamedQubit('a')
    assert_equivalent_op_tree([cirq.X(a)], [cirq.X(a)])
    assert_equivalent_op_tree(cirq.Circuit([cirq.X(a)]), [cirq.X(a)])
    assert_equivalent_op_tree(cirq.Circuit([cirq.X(a)], cirq.Moment()), [cirq.X(a)])
    with pytest.raises(AssertionError):
        assert_equivalent_op_tree([cirq.X(a)], [])