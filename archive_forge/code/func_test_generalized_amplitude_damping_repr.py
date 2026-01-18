import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_generalized_amplitude_damping_repr():
    cirq.testing.assert_equivalent_repr(cirq.GeneralizedAmplitudeDampingChannel(0.1, 0.3))