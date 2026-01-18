import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_noise_model():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    noise_model = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(p=0.01))
    simulator = cirq.Simulator(noise=noise_model)
    result = simulator.run(circuit, repetitions=100)
    assert 20 <= sum(result.measurements['q(0)'])[0] < 80