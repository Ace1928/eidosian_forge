from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_measured_channel():
    kc = cirq.KrausChannel(kraus_ops=(np.array([[1, 1], [1, 1]]) * 0.5, np.array([[1, -1], [-1, 1]]) * 0.5), key='m')
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), kc.on(q0))
    sim = cirq.Simulator(seed=0)
    results = sim.run(circuit, repetitions=100)
    assert results.histogram(key='m') == {0: 100}