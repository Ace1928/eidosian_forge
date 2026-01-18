import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_incorporate_result_not_view():
    tensor = np.zeros((2, 2))
    tensor2 = np.zeros((2, 2))
    buffer = np.empty_like(tensor)
    args = cirq.ApplyUnitaryArgs(tensor, buffer, [0])
    not_sub_args = cirq.ApplyUnitaryArgs(tensor2, buffer, [0])
    with pytest.raises(ValueError, match='view'):
        _incorporate_result_into_target(args, not_sub_args, tensor2)