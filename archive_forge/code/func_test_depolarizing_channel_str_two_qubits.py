import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_str_two_qubits():
    assert str(cirq.depolarize(0.3, n_qubits=2)) == 'depolarize(p=0.3,n_qubits=2)'