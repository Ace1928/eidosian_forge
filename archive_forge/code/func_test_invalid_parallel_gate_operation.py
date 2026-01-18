import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('gate, num_copies, qubits, error_msg', [(cirq.testing.SingleQubitGate(), 3, cirq.LineQubit.range(2), 'Wrong number of qubits'), (cirq.testing.SingleQubitGate(), 0, cirq.LineQubit.range(4), 'gate must be applied at least once'), (cirq.testing.SingleQubitGate(), 2, [cirq.NamedQubit('a'), cirq.NamedQubit('a')], 'Duplicate'), (cirq.testing.TwoQubitGate(), 2, cirq.LineQubit.range(4), 'must be a single qubit gate')])
def test_invalid_parallel_gate_operation(gate, num_copies, qubits, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        cirq.ParallelGate(gate, num_copies)(*qubits)