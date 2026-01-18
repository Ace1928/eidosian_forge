import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def assert_works(val):
    expected_outputs = [np.array([1, 1, -1, -1]).reshape((2, 2)), np.array([1, -1, 1, -1]).reshape((2, 2))]
    for axis in range(2):
        result = cirq.apply_unitary(val, cirq.ApplyUnitaryArgs(make_input(), buf, [axis]))
        np.testing.assert_allclose(result, expected_outputs[axis])