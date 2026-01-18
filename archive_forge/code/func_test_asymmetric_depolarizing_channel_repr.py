import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_asymmetric_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.AsymmetricDepolarizingChannel(0.1, 0.2, 0.3))