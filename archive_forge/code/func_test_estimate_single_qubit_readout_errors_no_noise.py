from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_single_qubit_readout_errors_no_noise():
    qubits = cirq.LineQubit.range(10)
    sampler = cirq.Simulator()
    repetitions = 1000
    result = cirq.estimate_single_qubit_readout_errors(sampler, qubits=qubits, repetitions=repetitions)
    assert result.zero_state_errors == {q: 0 for q in qubits}
    assert result.one_state_errors == {q: 0 for q in qubits}
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)