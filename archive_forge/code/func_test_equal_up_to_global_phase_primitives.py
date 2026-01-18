import numpy as np
import cirq
def test_equal_up_to_global_phase_primitives():
    assert cirq.equal_up_to_global_phase(1.0 + 1j, 1.0 + 1j, atol=1e-09)
    assert not cirq.equal_up_to_global_phase(2.0, 1.0 + 1j, atol=1e-09)
    assert cirq.equal_up_to_global_phase(1.0 + 1j, 1.0 - 1j, atol=1e-09)
    assert cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 1.0 + 0j, atol=1e-09)
    assert cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 1j, atol=1e-09)
    assert cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 1, atol=1e-09)
    assert not cirq.equal_up_to_global_phase(np.exp(1j * 3.3), 0, atol=1e-09)
    assert cirq.equal_up_to_global_phase(1j, 1 + 1e-10, atol=1e-09)
    assert not cirq.equal_up_to_global_phase(1j, 1 + 1e-10, atol=1e-11)
    assert cirq.equal_up_to_global_phase(1.0 + 0.1j, 1.0, atol=0.01)
    assert not cirq.equal_up_to_global_phase(1.0 + 0.1j, 1.0, atol=0.001)
    assert cirq.equal_up_to_global_phase(1.0 + 1j, np.sqrt(2) + 1e-08, atol=1e-07)
    assert not cirq.equal_up_to_global_phase(1.0 + 1j, np.sqrt(2) + 1e-07, atol=1e-08)
    assert cirq.equal_up_to_global_phase(1.0 + 1e-10j, 1.0, atol=1e-15)