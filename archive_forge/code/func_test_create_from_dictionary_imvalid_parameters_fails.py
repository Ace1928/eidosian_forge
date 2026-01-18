from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
def test_create_from_dictionary_imvalid_parameters_fails():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CZ(a, b))
    simulator = PhasedFSimEngineSimulator.create_from_dictionary({})
    with pytest.raises(ValueError, match='Missing parameters'):
        simulator.final_state_vector(circuit)
    with pytest.raises(ValueError, match='canonical order'):
        PhasedFSimEngineSimulator.create_from_dictionary(parameters={(b, a): {'theta': 0.6, 'phi': 0.2}})