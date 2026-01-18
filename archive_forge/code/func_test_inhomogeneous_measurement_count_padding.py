import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_inhomogeneous_measurement_count_padding():
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    sim = cirq.Simulator()
    c = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q) ** 0.2, cirq.measure(q, key=key)), use_repetition_ids=False, repeat_until=cirq.KeyCondition(key)))
    results = sim.run(c, repetitions=10)
    for i in range(10):
        assert np.sum(results.records['m'][i, :, :]) == 1