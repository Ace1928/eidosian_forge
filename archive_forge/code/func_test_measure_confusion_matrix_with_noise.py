import numpy as np
import cirq
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
@pytest.mark.parametrize('p0, p1', [(0, 0), (0.2, 0.4), (0.5, 0.5), (0.6, 0.3), (1.0, 1.0)])
def test_measure_confusion_matrix_with_noise(p0, p1):
    sampler = NoisySingleQubitReadoutSampler(p0, p1, seed=1234)
    num_qubits = 4
    qubits = cirq.LineQubit.range(num_qubits)
    expected_cm = get_expected_cm(num_qubits, p0, p1)
    qubits_small = qubits[:2]
    expected_cm_small = get_expected_cm(2, p0, p1)
    repetitions = 12000
    readout_cm = cirq.measure_confusion_matrix(sampler, qubits, repetitions=repetitions)
    assert readout_cm.repetitions == repetitions
    for q, expected in zip([None, qubits_small], [expected_cm, expected_cm_small]):
        np.testing.assert_allclose(readout_cm.confusion_matrix(q), expected, atol=0.01)
        np.testing.assert_allclose(readout_cm.confusion_matrix(q) @ readout_cm.correction_matrix(q), np.eye(expected.shape[0]), atol=0.01)
    readout_cm = cirq.measure_confusion_matrix(sampler, [[q] for q in qubits], repetitions=repetitions)
    assert readout_cm.repetitions == repetitions
    for q, expected in zip([None, qubits_small], [expected_cm, expected_cm_small]):
        np.testing.assert_allclose(readout_cm.confusion_matrix(q), expected, atol=0.01)
        np.testing.assert_allclose(readout_cm.confusion_matrix(q) @ readout_cm.correction_matrix(q), np.eye(expected.shape[0]), atol=0.01)
    qs = qubits_small
    circuit = cirq.Circuit(cirq.H.on_each(*qs), cirq.measure(*qs))
    reps = 100000
    sampled_result = cirq.get_state_histogram(sampler.run(circuit, repetitions=reps)) / reps
    expected_result = [1 / 4] * 4

    def l2norm(result: np.ndarray):
        return np.sum((expected_result - result) ** 2)
    corrected_result = readout_cm.apply(sampled_result, qs)
    assert l2norm(corrected_result) <= l2norm(sampled_result)