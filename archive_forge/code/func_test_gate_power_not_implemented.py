import math
import cirq
import numpy
import pytest
import cirq_ionq as ionq
@pytest.mark.parametrize('gate,power', [*[(ionq.GPIGate(phi=0.1), power) for power in INVALID_GATE_POWER], *[(ionq.GPI2Gate(phi=0.1), power) for power in INVALID_GATE_POWER], *[(ionq.MSGate(phi0=0.1, phi1=0.2), power) for power in INVALID_GATE_POWER]])
def test_gate_power_not_implemented(gate, power):
    """Tests that any power other than 1 and -1 is not implemented."""
    with pytest.raises(TypeError):
        _ = gate ** power