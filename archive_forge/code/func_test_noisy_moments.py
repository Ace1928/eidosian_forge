import pytest
import cirq
from cirq_pasqal import PasqalNoiseModel, PasqalDevice
from cirq.ops import NamedQubit
def test_noisy_moments():
    p_qubits = cirq.NamedQubit.range(2, prefix='q')
    p_device = PasqalDevice(qubits=p_qubits)
    noise_model = PasqalNoiseModel(p_device)
    circuit = cirq.Circuit()
    circuit.append(cirq.ops.CZ(p_qubits[0], p_qubits[1]))
    circuit.append(cirq.ops.Z(p_qubits[1]))
    p_circuit = cirq.Circuit(circuit)
    n_mts = []
    for moment in p_circuit._moments:
        n_mts.append(noise_model.noisy_moment(moment, p_qubits))
    assert n_mts == [[cirq.ops.CZ.on(NamedQubit('q0'), NamedQubit('q1')), cirq.depolarize(p=0.03).on(NamedQubit('q0')), cirq.depolarize(p=0.03).on(NamedQubit('q1'))], [cirq.ops.Z.on(NamedQubit('q1')), cirq.depolarize(p=0.01).on(NamedQubit('q1'))]]