import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_density_matrix_trial_result_qid_shape():
    q0, q1 = cirq.LineQubit.range(2)
    final_simulator_state = cirq.DensityMatrixSimulationState(initial_state=np.ones((4, 4)) / 4, qubits=[q0, q1])
    assert cirq.qid_shape(cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state)) == (2, 2)
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    final_simulator_state = cirq.DensityMatrixSimulationState(initial_state=np.ones((12, 12)) / 12, qubits=[q0, q1])
    assert cirq.qid_shape(cirq.DensityMatrixTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state)) == (3, 4)