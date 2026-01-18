import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
@pytest.mark.parametrize('reps,expected', [(1000, 2.25), (16000, 2.9), (64000, 4.6), (128000, 7.4)])
def test_estimate_run_time_vary_reps(reps, expected):
    """Test various run times.
    Values taken from Weber November 2021."""
    qubits = cirq.GridQubit.rect(2, 5)
    circuit = cirq.testing.random_circuit(qubits, n_moments=10, op_density=1.0)
    runtime = runtime_estimator.estimate_run_time(circuit, repetitions=reps)
    _assert_about_equal(runtime, expected)