from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_infers_other_methods():
    q = cirq.LineQubit(0)

    class NoiseModelWithNoisyMomentListMethod(cirq.NoiseModel):

        def noisy_moments(self, moments, system_qubits):
            result = []
            for moment in moments:
                if moment.operations:
                    result.append(cirq.X(moment.operations[0].qubits[0]).with_tags(ops.VirtualTag()))
                else:
                    result.append([])
            return result
    a = NoiseModelWithNoisyMomentListMethod()
    assert_equivalent_op_tree(a.noisy_operation(cirq.H(q)), cirq.X(q).with_tags(ops.VirtualTag()))
    assert_equivalent_op_tree(a.noisy_moment(cirq.Moment([cirq.H(q)]), [q]), cirq.X(q).with_tags(ops.VirtualTag()))
    assert_equivalent_op_tree_sequence(a.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]), [[], cirq.X(q).with_tags(ops.VirtualTag())])

    class NoiseModelWithNoisyMomentMethod(cirq.NoiseModel):

        def noisy_moment(self, moment, system_qubits):
            return [y.with_tags(ops.VirtualTag()) for y in cirq.Y.on_each(*moment.qubits)]
    b = NoiseModelWithNoisyMomentMethod()
    assert_equivalent_op_tree(b.noisy_operation(cirq.H(q)), cirq.Y(q).with_tags(ops.VirtualTag()))
    assert_equivalent_op_tree(b.noisy_moment(cirq.Moment([cirq.H(q)]), [q]), cirq.Y(q).with_tags(ops.VirtualTag()))
    assert_equivalent_op_tree_sequence(b.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]), [[], cirq.Y(q).with_tags(ops.VirtualTag())])

    class NoiseModelWithNoisyOperationMethod(cirq.NoiseModel):

        def noisy_operation(self, operation: 'cirq.Operation'):
            return cirq.Z(operation.qubits[0]).with_tags(ops.VirtualTag())
    c = NoiseModelWithNoisyOperationMethod()
    assert_equivalent_op_tree(c.noisy_operation(cirq.H(q)), cirq.Z(q).with_tags(ops.VirtualTag()))
    assert_equivalent_op_tree(c.noisy_moment(cirq.Moment([cirq.H(q)]), [q]), cirq.Z(q).with_tags(ops.VirtualTag()))
    assert_equivalent_op_tree_sequence(c.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]), [[], cirq.Z(q).with_tags(ops.VirtualTag())])