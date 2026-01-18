import itertools
from typing import Sequence
import numpy as np
import pytest
import cirq
def sample_noisy_bitstrings(circuit: cirq.Circuit, qubit_order: Sequence[cirq.Qid], depolarization: float, repetitions: int) -> np.ndarray:
    assert 0 <= depolarization <= 1
    dim = np.prod(circuit.qid_shape(), dtype=np.int64)
    n_incoherent = int(depolarization * repetitions)
    n_coherent = repetitions - n_incoherent
    incoherent_samples = np.random.randint(dim, size=n_incoherent)
    circuit_with_measurements = cirq.Circuit(circuit, cirq.measure(*qubit_order, key='m'))
    r = cirq.sample(circuit_with_measurements, repetitions=n_coherent)
    coherent_samples = r.data['m'].to_numpy()
    return np.concatenate((coherent_samples, incoherent_samples))