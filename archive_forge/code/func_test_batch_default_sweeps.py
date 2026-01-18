from typing import List
import pytest
import sympy
import numpy as np
import cirq
import cirq_google as cg
def test_batch_default_sweeps():
    sampler = cg.ValidatingSampler()
    q = cirq.GridQubit(2, 2)
    circuits = [cirq.Circuit(cirq.X(q), cirq.measure(q, key='m')), cirq.Circuit(cirq.measure(q, key='m2'))]
    results = sampler.run_batch(circuits, None, repetitions=100)
    assert np.all(results[0][0].measurements['m'] == 1)
    assert np.all(results[1][0].measurements['m2'] == 0)