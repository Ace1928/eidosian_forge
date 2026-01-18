from typing import Sequence
import pytest
import numpy as np
import cirq
def test_estimate_parallel_readout_errors_with_noise():
    qubits = cirq.LineQubit.range(5)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    repetitions = 1000
    result = cirq.estimate_parallel_single_qubit_readout_errors(sampler, qubits=qubits, repetitions=repetitions, trials=40)
    for error in result.one_state_errors.values():
        assert 0.17 < error < 0.23
    for error in result.zero_state_errors.values():
        assert 0.07 < error < 0.13
    assert result.repetitions == repetitions
    assert isinstance(result.timestamp, float)