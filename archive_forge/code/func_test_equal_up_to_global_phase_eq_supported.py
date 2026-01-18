import numpy as np
import cirq
def test_equal_up_to_global_phase_eq_supported():
    assert cirq.equal_up_to_global_phase(A(0.1 + 0j), A(0.1j), atol=0.01)
    assert not cirq.equal_up_to_global_phase(A(0.0 + 0j), A(0.1j), atol=0.0)
    assert not cirq.equal_up_to_global_phase(A(0.0 + 0j), 0.1j, atol=0.0)
    assert cirq.equal_up_to_global_phase(B(0j), 1e-08j, atol=1e-08)
    assert cirq.equal_up_to_global_phase(1e-08j, B(0j), atol=1e-08)
    assert not cirq.equal_up_to_global_phase(1e-08j, B(0.0 + 0j), atol=1e-10)
    assert cirq.equal_up_to_global_phase(A(0.1), A(0.1j), atol=0.01)
    assert not cirq.equal_up_to_global_phase(1e-08j, B(0.0), atol=1e-10)