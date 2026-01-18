import numpy as np
import pytest
import cirq
import sympy
def test_compatible_measurement():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(qs, key='key'), cirq.X.on_each(qs), cirq.measure(qs, key='key'))
    sim = cirq.ClassicalStateSimulator()
    res = sim.run(c, repetitions=3).records
    np.testing.assert_equal(res['key'], np.array([[[0, 0], [1, 1]]] * 3, dtype=np.uint8))