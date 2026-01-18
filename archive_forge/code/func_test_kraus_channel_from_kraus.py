import cirq
import numpy as np
import pytest
def test_kraus_channel_from_kraus():
    q0 = cirq.LineQubit(0)
    ops = [np.array([[1, 1], [1, 1]]) * 0.5, np.array([[1, -1], [-1, 1]]) * 0.5]
    x_meas = cirq.KrausChannel(ops, key='x_meas')
    assert cirq.measurement_key_name(x_meas) == 'x_meas'
    circuit = cirq.Circuit(cirq.H(q0), x_meas.on(q0))
    sim = cirq.Simulator(seed=0)
    results = sim.simulate(circuit)
    assert 'x_meas' in results.measurements
    assert results.measurements['x_meas'] == 0