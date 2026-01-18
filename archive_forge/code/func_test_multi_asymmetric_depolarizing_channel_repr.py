import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_multi_asymmetric_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.AsymmetricDepolarizingChannel(error_probabilities={'II': 0.8, 'XX': 0.2}))