import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_amplitude_damping_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.amplitude_damp(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.amplitude_damp(1.1)