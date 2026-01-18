import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_apply_two_qubits():
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.DepolarizingChannel(p=0.1, n_qubits=2)
    op(q0, q1)