import pytest
import sympy
import cirq
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
def test_consistent_protocols():
    gate = coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0, rise_time=cirq.Duration(nanos=18))
    cirq.testing.assert_implements_consistent_protocols(gate, setup_code='import cirq\nimport numpy as np\nimport sympy\nimport cirq_google', qubit_count=2, ignore_decompose_to_default_gateset=True)
    assert gate.num_qubits() == 2