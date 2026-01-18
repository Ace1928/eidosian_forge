from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_measure_gates():
    q00, q01, q10, q11 = cirq.GridQubit.rect(2, 2)
    qubits = [q00, q01, q10, q11]
    props = sample_noise_properties(qubits, [(q00, q01), (q01, q00), (q10, q11), (q11, q10), (q00, q10), (q10, q00), (q01, q11), (q11, q01)])
    model = NoiseModelFromGoogleNoiseProperties(props)
    op = cirq.measure(*qubits, key='m')
    circuit = cirq.Circuit(cirq.measure(*qubits, key='m'))
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 2
    assert len(noisy_circuit.moments[0].operations) == 4
    for q in qubits:
        op = noisy_circuit.moments[0].operation_at(q)
        assert isinstance(op.gate, cirq.GeneralizedAmplitudeDampingChannel), q
        assert np.isclose(op.gate.p, 0.9090909), q
        assert np.isclose(op.gate.gamma, 0.011), q
    assert len(noisy_circuit.moments[1].operations) == 1
    assert noisy_circuit.moments[1] == circuit.moments[0]