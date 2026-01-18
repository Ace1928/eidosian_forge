from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_measured_mixture():
    mm = cirq.MixedUnitaryChannel(mixture=((0.5, np.array([[1, 0], [0, 1]])), (0.5, np.array([[0, 1], [1, 0]]))), key='flip')
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(mm.on(q0), cirq.measure(q0, key='m'))
    sim = cirq.Simulator(seed=0)
    results = sim.run(circuit, repetitions=100)
    assert results.histogram(key='flip') == results.histogram(key='m')