import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_multi_asymmetric_depolarizing_eq():
    a = cirq.asymmetric_depolarize(error_probabilities={'I': 0.8, 'X': 0.2})
    b = cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})
    assert not cirq.approx_eq(a, b)
    a = cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})
    b = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'XX': 1 / 3})
    assert not cirq.approx_eq(a, b)
    a = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'ZZ': 1 / 3})
    b = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'XX': 1 / 3})
    assert not cirq.approx_eq(a, b)
    a = cirq.asymmetric_depolarize(0.1, 0.2)
    b = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'XX': 1 / 3})
    assert not cirq.approx_eq(a, b)
    a = cirq.asymmetric_depolarize(error_probabilities={'II': 0.667, 'XX': 0.333})
    b = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'XX': 1 / 3})
    assert cirq.approx_eq(a, b, atol=0.001)
    a = cirq.asymmetric_depolarize(error_probabilities={'II': 0.667, 'XX': 0.333})
    b = cirq.asymmetric_depolarize(error_probabilities={'XX': 1 / 3, 'II': 2 / 3})
    assert cirq.approx_eq(a, b, atol=0.001)