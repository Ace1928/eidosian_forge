from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
def test_aqt_sampler_empty_circuit():
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    _, _qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = True
    circuit = cirq.Circuit()
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    with pytest.raises(RuntimeError):
        _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)