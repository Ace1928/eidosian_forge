from typing import List
import pytest
import sympy
import numpy as np
import cirq
import cirq_google as cg
def test_device_validation():
    sampler = cg.ValidatingSampler(device=cg.Sycamore23, validator=lambda c, s, r: True, sampler=cirq.Simulator())
    q = cirq.GridQubit(5, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    results = sampler.run_sweep(circuit, sweep, repetitions=100)
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)
    q = cirq.GridQubit(2, 2)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    with pytest.raises(ValueError, match='Qubit not on device'):
        results = sampler.run_sweep(circuit, sweep, repetitions=100)