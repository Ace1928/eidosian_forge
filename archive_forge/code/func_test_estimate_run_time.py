import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
@pytest.mark.parametrize('depth, width, reps, expected', [(10, 80, 32000, 3.7), (10, 160, 32000, 4.5), (10, 320, 32000, 6.6), (10, 10, 32000, 3.5), (20, 10, 32000, 4.6), (30, 10, 32000, 5.9), (40, 10, 32000, 7.7), (50, 10, 32000, 9.4), (10, 10, 256000, 11.4), (40, 40, 256000, 26.8), (40, 160, 256000, 32.8), (40, 80, 32000, 12.1), (2, 40, 256000, 11.3), (2, 160, 256000, 11.4), (2, 640, 256000, 13.3), (2, 1280, 256000, 16.5), (2, 2560, 256000, 23.5), (10, 160, 256000, 18.2), (20, 160, 256000, 24.7), (30, 160, 256000, 30.8), (10, 1280, 256000, 38.6), (10, 1280, 1000, 18.7), (10, 1280, 256000, 38.6)])
def test_estimate_run_time(depth, width, reps, expected):
    """Test various run times.
    Values taken from Weber November 2021."""
    qubits = cirq.GridQubit.rect(8, 8)
    circuit = cirq.testing.random_circuit(qubits[:depth], n_moments=width, op_density=1.0)
    runtime = runtime_estimator.estimate_run_time(circuit, repetitions=reps)
    _assert_about_equal(runtime, expected)