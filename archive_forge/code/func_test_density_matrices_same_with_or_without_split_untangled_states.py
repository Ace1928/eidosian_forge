import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_density_matrices_same_with_or_without_split_untangled_states():
    sim = cirq.DensityMatrixSimulator(split_untangled_states=False)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX.on(q0, q1), cirq.reset(q1))
    result1 = sim.simulate(circuit).final_density_matrix
    sim = cirq.DensityMatrixSimulator()
    result2 = sim.simulate(circuit).final_density_matrix
    assert np.allclose(result1, result2)