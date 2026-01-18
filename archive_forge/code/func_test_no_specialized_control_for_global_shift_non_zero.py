import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('gate, specialized_type', [(cirq.ZPowGate(global_shift=-0.5, exponent=0.5), cirq.CZPowGate), (cirq.CZPowGate(global_shift=-0.5, exponent=0.5), cirq.CCZPowGate), (cirq.XPowGate(global_shift=-0.5, exponent=0.5), cirq.CXPowGate), (cirq.CXPowGate(global_shift=-0.5, exponent=0.5), cirq.CCXPowGate)])
def test_no_specialized_control_for_global_shift_non_zero(gate, specialized_type):
    assert not isinstance(gate.controlled(), specialized_type)