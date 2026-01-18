from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_parallel_readout_errors_bad_bit_string():
    qubits = cirq.LineQubit.range(4)
    with pytest.raises(ValueError, match='but was None'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(cirq.ZerosSampler(), qubits=qubits, repetitions=1000, trials=35, trials_per_batch=10, bit_strings=[[1] * 4])
    with pytest.raises(ValueError, match='0 or 1'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(cirq.ZerosSampler(), qubits=qubits, repetitions=1000, trials=2, bit_strings=np.array([[12, 47, 2, -4], [0.1, 7, 0, 0]]))