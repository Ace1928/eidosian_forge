import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_density_matrix_trial_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    dtype = np.complex64
    final_simulator_state = cirq.DensityMatrixSimulationState(available_buffer=[], prng=np.random.RandomState(0), qubits=[q0], initial_state=np.ones((2, 2), dtype=dtype) * 0.5, dtype=dtype)
    result = cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state)
    fake_printer = cirq.testing.FakePrinter()
    result._repr_pretty_(fake_printer, cycle=False)
    result_no_whitespace = fake_printer.text_pretty.replace('\n', '').replace(' ', '')
    assert result_no_whitespace == 'measurements:(nomeasurements)qubits:(cirq.LineQubit(0),)finaldensitymatrix:[[0.5+0.j0.5+0.j][0.5+0.j0.5+0.j]]'
    cirq.testing.assert_repr_pretty(result, 'cirq.DensityMatrixTrialResult(...)', cycle=True)