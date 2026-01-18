import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('split', [True, False])
def test_measurement_preserves_phase(split: bool):
    c1, c2, t = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(t), cirq.measure(t, key='t'), cirq.CZ(c1, c2).with_classical_controls('t'), cirq.reset(t))
    simulator = cirq.Simulator(split_untangled_states=split)
    for _ in range(20):
        result = simulator.simulate(circuit, initial_state=(1, 1, 1), qubit_order=(c1, c2, t))
        assert result.dirac_notation() == '|110⟩'