import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_bad_error_probabilities_gate():
    with pytest.raises(ValueError, match='AB is not made solely of I, X, Y, Z.'):
        cirq.asymmetric_depolarize(error_probabilities={'AB': 1.0})
    with pytest.raises(ValueError, match='Y must have 2 Pauli gates.'):
        cirq.asymmetric_depolarize(error_probabilities={'IX': 0.8, 'Y': 0.2})