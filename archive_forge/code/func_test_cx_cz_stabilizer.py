import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('gate', [cirq.CX, cirq.CZ])
def test_cx_cz_stabilizer(gate):
    assert cirq.has_stabilizer_effect(gate)
    assert not cirq.has_stabilizer_effect(gate ** 0.5)
    assert cirq.has_stabilizer_effect(gate ** 0)
    assert not cirq.has_stabilizer_effect(gate ** (-0.5))
    assert cirq.has_stabilizer_effect(gate ** 4)
    assert not cirq.has_stabilizer_effect(gate ** 1.2)
    foo = sympy.Symbol('foo')
    assert not cirq.has_stabilizer_effect(gate ** foo)