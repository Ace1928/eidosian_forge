import pytest
import pandas as pd
import sympy
import cirq
import cirq.experiments.t2_decay_experiment as t2
@pytest.mark.parametrize('experiment_type', [t2.ExperimentType.RAMSEY, t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG])
def test_all_on_results(experiment_type):
    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = t2.t2_decay(sampler=cirq.Simulator(), qubit=cirq.GridQubit(0, 0), num_points=4, repetitions=500, min_delay=cirq.Duration(nanos=100), max_delay=cirq.Duration(micros=1), num_pulses=pulses, experiment_type=experiment_type)
    assert (results.expectation_pauli_y['value'] == -1.0).all()
    assert (abs(results.expectation_pauli_x['value']) < 0.2).all()