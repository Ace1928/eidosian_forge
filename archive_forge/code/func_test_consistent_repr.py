from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_consistent_repr():
    q0, q1 = cirq.LineQubit.range(2)
    test_props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    cirq.testing.assert_equivalent_repr(test_props, setup_code='import cirq, cirq_google\nimport numpy as np')