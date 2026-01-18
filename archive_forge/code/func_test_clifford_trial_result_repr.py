import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_trial_result_repr():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.StabilizerChFormSimulationState(qubits=[q0])
    assert repr(cirq.CliffordTrialResult(params=cirq.ParamResolver({}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state)) == "cirq.SimulationTrialResult(params=cirq.ParamResolver({}), measurements={'m': array([[1]])}, final_simulator_state=cirq.StabilizerChFormSimulationState(initial_state=StabilizerStateChForm(num_qubits=1), qubits=(cirq.LineQubit(0),), classical_data=cirq.ClassicalDataDictionaryStore()))"