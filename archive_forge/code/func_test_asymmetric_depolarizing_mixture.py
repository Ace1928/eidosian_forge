import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_asymmetric_depolarizing_mixture():
    d = cirq.asymmetric_depolarize(0.1, 0.2, 0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.4, np.eye(2)), (0.1, X), (0.2, Y), (0.3, Z)))
    assert cirq.has_mixture(d)