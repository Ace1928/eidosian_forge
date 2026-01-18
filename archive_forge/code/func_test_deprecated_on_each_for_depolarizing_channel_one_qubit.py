import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_deprecated_on_each_for_depolarizing_channel_one_qubit():
    q0 = cirq.LineQubit.range(1)
    op = cirq.DepolarizingChannel(p=0.1, n_qubits=1)
    op.on_each(q0)
    op.on_each([q0])
    with pytest.raises(ValueError, match='Gate was called with type different than Qid'):
        op.on_each('bogus object')