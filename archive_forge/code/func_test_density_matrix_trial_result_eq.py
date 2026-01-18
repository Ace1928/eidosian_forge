import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_density_matrix_trial_result_eq():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.DensityMatrixSimulationState(initial_state=np.ones((2, 2)) * 0.5, qubits=[q0])
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state), cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state))
    eq.add_equality_group(cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={}, final_simulator_state=final_simulator_state))
    eq.add_equality_group(cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state))