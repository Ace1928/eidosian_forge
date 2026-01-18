from typing import List
import pytest
import sympy
import numpy as np
import cirq
import cirq_google as cg
def test_batch_validation():
    sampler = cg.ValidatingSampler(device=cirq.UNCONSTRAINED_DEVICE, validator=_batch_size_less_than_two, sampler=cirq.Simulator())
    q = cirq.GridQubit(2, 2)
    circuits = [cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m')), cirq.Circuit(cirq.X(q) ** sympy.Symbol('x'), cirq.measure(q, key='m2'))]
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1])]
    results = sampler.run_batch(circuits, sweeps, repetitions=100)
    assert np.all(results[0][0].measurements['m'] == 1)
    assert np.all(results[0][1].measurements['m'] == 0)
    assert np.all(results[1][0].measurements['m2'] == 0)
    assert np.all(results[1][1].measurements['m2'] == 1)
    circuits = [cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m')), cirq.Circuit(cirq.X(q) ** sympy.Symbol('x'), cirq.measure(q, key='m2')), cirq.Circuit(cirq.measure(q, key='m3'))]
    sweeps = [cirq.Points(key='t', points=[1, 0]), cirq.Points(key='x', points=[0, 1]), {}]
    with pytest.raises(ValueError, match='Too many batches'):
        results = sampler.run_batch(circuits, sweeps, repetitions=100)