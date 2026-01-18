from typing import cast
import numpy as np
import pytest
import cirq
def test_act_on_qutrit():
    a, b = [cirq.LineQid(3, dimension=3), cirq.LineQid(1, dimension=3)]
    m = cirq.measure(a, b, key='out', invert_mask=(True,), confusion_map={(1,): np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])})
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(3, 3, 3, 3, 3)), qubits=cirq.LineQid.range(5, dimension=3), prng=np.random.RandomState(), initial_state=cirq.one_hot(index=(0, 2, 0, 2, 0), shape=(3, 3, 3, 3, 3), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [2, 0]}
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(3, 3, 3, 3, 3)), qubits=cirq.LineQid.range(5, dimension=3), prng=np.random.RandomState(), initial_state=cirq.one_hot(index=(0, 1, 0, 2, 0), shape=(3, 3, 3, 3, 3), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [2, 2]}
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(3, 3, 3, 3, 3)), qubits=cirq.LineQid.range(5, dimension=3), prng=np.random.RandomState(), initial_state=cirq.one_hot(index=(0, 2, 0, 1, 0), shape=(3, 3, 3, 3, 3), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}