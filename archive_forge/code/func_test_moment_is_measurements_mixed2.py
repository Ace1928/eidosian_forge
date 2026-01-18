from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_moment_is_measurements_mixed2():
    q = cirq.LineQubit.range(2)
    circ = cirq.Circuit([cirq.X(q[0]), cirq.X(q[1]), cirq.Z(q[0]), cirq.measure(q[1], key='z')])
    assert not validate_all_measurements(circ[0])
    with pytest.raises(ValueError) as e:
        validate_all_measurements(circ[1])
    assert e.match('.*must be homogeneous: all measurements.*')