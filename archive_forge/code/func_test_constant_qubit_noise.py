from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_constant_qubit_noise():
    a, b, c = cirq.LineQubit.range(3)
    damp = cirq.amplitude_damp(0.5)
    damp_all = cirq.ConstantQubitNoiseModel(damp)
    actual = damp_all.noisy_moments([cirq.Moment([cirq.X(a)]), cirq.Moment()], [a, b, c])
    expected = [[cirq.Moment([cirq.X(a)]), cirq.Moment((d.with_tags(ops.VirtualTag()) for d in [damp(a), damp(b), damp(c)]))], [cirq.Moment(), cirq.Moment((d.with_tags(ops.VirtualTag()) for d in [damp(a), damp(b), damp(c)]))]]
    assert actual == expected
    cirq.testing.assert_equivalent_repr(damp_all)
    with pytest.raises(ValueError, match='num_qubits'):
        _ = cirq.ConstantQubitNoiseModel(cirq.CNOT ** 0.01)