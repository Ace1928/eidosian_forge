import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_default_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize()
    assert d.p_i == 1.0
    assert d.p_x == 0.0
    assert d.p_y == 0.0
    assert d.p_z == 0.0
    assert d.num_qubits() == 1