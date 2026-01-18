import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_bit_flip_mixture():
    d = cirq.bit_flip(0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.7, np.eye(2)), (0.3, X)))
    assert cirq.has_mixture(d)