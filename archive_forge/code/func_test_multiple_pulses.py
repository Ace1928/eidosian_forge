import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
def test_multiple_pulses():
    results = t2.t2_decay(sampler=cirq.DensityMatrixSimulator(noise=cirq.amplitude_damp(1)), qubit=cirq.GridQubit(0, 0), num_points=4, repetitions=10, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1), experiment_type=t2.ExperimentType.CPMG, num_pulses=[1, 2, 3, 4], delay_sweep=cirq.Points('delay_ns', [1.0, 10.0, 100.0, 1000.0, 10000.0]))
    data = [[1.0, 1, 10, 0], [1.0, 2, 10, 0], [1.0, 3, 10, 0], [1.0, 4, 10, 0], [10.0, 1, 10, 0], [10.0, 2, 10, 0], [10.0, 3, 10, 0], [10.0, 4, 10, 0], [100.0, 1, 10, 0], [100.0, 2, 10, 0], [100.0, 3, 10, 0], [100.0, 4, 10, 0], [1000.0, 1, 10, 0], [1000.0, 2, 10, 0], [1000.0, 3, 10, 0], [1000.0, 4, 10, 0], [10000.0, 1, 10, 0], [10000.0, 2, 10, 0], [10000.0, 3, 10, 0], [10000.0, 4, 10, 0]]
    assert results == cirq.experiments.T2DecayResult(x_basis_data=pd.DataFrame(columns=['delay_ns', 'num_pulses', 0, 1], index=range(20), data=data), y_basis_data=pd.DataFrame(columns=['delay_ns', 'num_pulses', 0, 1], index=range(20), data=data))
    expected = pd.DataFrame(columns=['delay_ns', 'num_pulses', 'value'], index=range(20), data=[[1.0, 1, -1.0], [1.0, 2, -1.0], [1.0, 3, -1.0], [1.0, 4, -1.0], [10.0, 1, -1.0], [10.0, 2, -1.0], [10.0, 3, -1.0], [10.0, 4, -1.0], [100.0, 1, -1.0], [100.0, 2, -1.0], [100.0, 3, -1.0], [100.0, 4, -1.0], [1000.0, 1, -1.0], [1000.0, 2, -1.0], [1000.0, 3, -1.0], [1000.0, 4, -1.0], [10000.0, 1, -1.0], [10000.0, 2, -1.0], [10000.0, 3, -1.0], [10000.0, 4, -1.0]])
    assert results.expectation_pauli_x.equals(expected)