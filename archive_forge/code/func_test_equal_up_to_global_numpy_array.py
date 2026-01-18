import numpy as np
import cirq
def test_equal_up_to_global_numpy_array():
    assert cirq.equal_up_to_global_phase(np.asarray([1j, 1j]), np.asarray([1, 1], dtype=np.complex64))
    assert not cirq.equal_up_to_global_phase(np.asarray([1j, -1j]), np.asarray([1, 1], dtype=np.complex64))
    assert cirq.equal_up_to_global_phase(np.asarray([]), np.asarray([]))
    assert cirq.equal_up_to_global_phase(np.asarray([[]]), np.asarray([[]]))