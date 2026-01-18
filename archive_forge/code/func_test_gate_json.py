import math
import cirq
import numpy
import pytest
import cirq_ionq as ionq
@pytest.mark.parametrize('gate', [ionq.GPIGate(phi=0.1), ionq.GPI2Gate(phi=0.2), ionq.MSGate(phi0=0.1, phi1=0.2)])
def test_gate_json(gate):
    g_json = cirq.to_json(gate)
    assert cirq.read_json(json_text=g_json) == gate