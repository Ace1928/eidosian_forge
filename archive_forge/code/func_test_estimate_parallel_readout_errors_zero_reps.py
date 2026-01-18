from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_parallel_readout_errors_zero_reps():
    qubits = cirq.LineQubit.range(10)
    with pytest.raises(ValueError, match='non-zero repetition'):
        _ = cirq.estimate_parallel_single_qubit_readout_errors(cirq.ZerosSampler(), qubits=qubits, repetitions=0, trials=35, trials_per_batch=10)