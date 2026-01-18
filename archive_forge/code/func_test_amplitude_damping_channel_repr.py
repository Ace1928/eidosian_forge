import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_amplitude_damping_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.AmplitudeDampingChannel(0.3))