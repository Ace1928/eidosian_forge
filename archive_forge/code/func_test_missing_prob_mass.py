import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_missing_prob_mass():
    with pytest.raises(ValueError, match='Probabilities do not add up to 1'):
        cirq.asymmetric_depolarize(error_probabilities={'X': 0.1, 'I': 0.2})
    d = cirq.asymmetric_depolarize(error_probabilities={'X': 0.1})
    np.testing.assert_almost_equal(d.error_probabilities['I'], 0.9)