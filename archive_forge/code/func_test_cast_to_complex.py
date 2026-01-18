import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_cast_to_complex():
    y0 = cirq.PauliString({cirq.LineQubit(0): cirq.Y})
    state = 0.5 * np.eye(2)
    args = cirq.ApplyUnitaryArgs(target_tensor=state, available_buffer=np.zeros_like(state), axes=(0,))
    with pytest.raises(np.ComplexWarning, match='Casting complex values to real discards the imaginary part'):
        cirq.apply_unitary(y0, args)