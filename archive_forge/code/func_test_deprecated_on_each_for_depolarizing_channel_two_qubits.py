import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_deprecated_on_each_for_depolarizing_channel_two_qubits():
    q0, q1, q2, q3, q4, q5 = cirq.LineQubit.range(6)
    op = cirq.DepolarizingChannel(p=0.1, n_qubits=2)
    op.on_each([(q0, q1)])
    op.on_each([(q0, q1), (q2, q3)])
    op.on_each(zip([q0, q2, q4], [q1, q3, q5]))
    op.on_each((q0, q1))
    op.on_each([q0, q1])
    with pytest.raises(ValueError, match='Inputs to multi-qubit gates must be Sequence'):
        op.on_each(q0, q1)
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        op.on_each([('bogus object 0', 'bogus object 1')])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        op.on_each(['01'])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        op.on_each([(False, None)])