import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_asymmetric_depolarizing_channel_str():
    assert str(cirq.asymmetric_depolarize(0.1, 0.2, 0.3)) == "asymmetric_depolarize(error_probabilities={'I': 0.3999999999999999, " + "'X': 0.1, 'Y': 0.2, 'Z': 0.3})"