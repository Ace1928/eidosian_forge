from typing import cast
import numpy as np
import pytest
import cirq
def test_act_on_state_vector():
    a, b = [cirq.LineQubit(3), cirq.LineQubit(1)]
    m = cirq.measure(a, b, key='out', invert_mask=(True,), confusion_map={(1,): np.array([[0, 1], [1, 0]])})
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(2, 2, 2, 2, 2)), qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=cirq.one_hot(shape=(2, 2, 2, 2, 2), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 1]}
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(2, 2, 2, 2, 2)), qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=cirq.one_hot(index=(0, 1, 0, 0, 0), shape=(2, 2, 2, 2, 2), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 0]}
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(2, 2, 2, 2, 2)), qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=cirq.one_hot(index=(0, 1, 0, 1, 0), shape=(2, 2, 2, 2, 2), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(m, args)
    datastore = cast(cirq.ClassicalDataDictionaryStore, args.classical_data)
    out = cirq.MeasurementKey('out')
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0)]
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0), (0, 0)]