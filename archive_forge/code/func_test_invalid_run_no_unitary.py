import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_invalid_run_no_unitary():

    class NoUnitary(cirq.testing.SingleQubitGate):
        pass
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator()
    circuit = cirq.Circuit(NoUnitary()(q0))
    circuit.append([cirq.measure(q0, key='meas')])
    with pytest.raises(TypeError, match='unitary'):
        simulator.run(circuit)