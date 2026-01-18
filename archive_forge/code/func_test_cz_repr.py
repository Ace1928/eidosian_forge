import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_cz_repr():
    assert repr(cirq.CZ) == 'cirq.CZ'
    assert repr(cirq.CZ ** 0.5) == '(cirq.CZ**0.5)'
    assert repr(cirq.CZ ** (-0.25)) == '(cirq.CZ**-0.25)'