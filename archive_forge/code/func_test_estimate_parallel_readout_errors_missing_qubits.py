from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_parallel_readout_errors_missing_qubits():
    qubits = cirq.LineQubit.range(4)
    result = cirq.estimate_parallel_single_qubit_readout_errors(cirq.ZerosSampler(), qubits=qubits, repetitions=2000, trials=1, bit_strings=np.array([[0] * 4]))
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert all((np.isnan(result.one_state_errors[q]) for q in qubits))
    assert result.repetitions == 2000
    assert isinstance(result.timestamp, float)