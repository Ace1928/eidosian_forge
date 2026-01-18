import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('p, gamma', ((-0.1, 0.0), (0.0, -0.1), (0.1, -0.1), (-0.1, 0.1)))
def test_generalized_amplitude_damping_channel_negative_probability(p, gamma):
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.generalized_amplitude_damp(p, gamma)