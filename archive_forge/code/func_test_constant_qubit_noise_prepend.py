from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_constant_qubit_noise_prepend():
    a, b, c = cirq.LineQubit.range(3)
    damp = cirq.amplitude_damp(0.5)
    damp_all = cirq.ConstantQubitNoiseModel(damp, prepend=True)
    actual = damp_all.noisy_moments([cirq.Moment([cirq.X(a)]), cirq.Moment()], [a, b, c])
    expected = [[cirq.Moment((d.with_tags(ops.VirtualTag()) for d in [damp(a), damp(b), damp(c)])), cirq.Moment([cirq.X(a)])], [cirq.Moment((d.with_tags(ops.VirtualTag()) for d in [damp(a), damp(b), damp(c)])), cirq.Moment()]]
    assert actual == expected
    cirq.testing.assert_equivalent_repr(damp_all)