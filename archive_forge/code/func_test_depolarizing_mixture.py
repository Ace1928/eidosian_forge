import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_mixture():
    d = cirq.depolarize(0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.7, np.eye(2)), (0.1, X), (0.1, Y), (0.1, Z)))
    assert cirq.has_mixture(d)