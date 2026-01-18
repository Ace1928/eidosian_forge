import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
@pytest.mark.parametrize('gamma', [0.01, 0.05, 0.1])
def test_noise_model_discrete(gamma):
    results = cirq.experiments.t1_decay(sampler=cirq.DensityMatrixSimulator(noise=cirq.NoiseModel.from_noise_model_like(cirq.amplitude_damp(gamma))), qubit=cirq.GridQubit(0, 0), num_points=4, repetitions=100, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1))
    data = results.data
    probs = data['true_count'] / (data['true_count'] + data['false_count'])
    np.testing.assert_allclose(probs, np.mean(probs), atol=0.2)