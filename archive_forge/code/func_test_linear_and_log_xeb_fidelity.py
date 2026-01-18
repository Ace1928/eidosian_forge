import itertools
from typing import Sequence
import numpy as np
import pytest
import cirq
def test_linear_and_log_xeb_fidelity():
    prng_state = np.random.get_state()
    np.random.seed(0)
    depolarization = 0.5
    fs_log = []
    fs_lin = []
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(circuit, qubits, depolarization=depolarization, repetitions=5000)
        f_log = cirq.log_xeb_fidelity(circuit, bitstrings, qubits)
        f_lin = cirq.linear_xeb_fidelity(circuit, bitstrings, qubits)
        fs_log.append(f_log)
        fs_lin.append(f_lin)
    assert np.isclose(np.mean(fs_log), 1 - depolarization, atol=0.01)
    assert np.isclose(np.mean(fs_lin), 1 - depolarization, atol=0.09)
    np.random.set_state(prng_state)