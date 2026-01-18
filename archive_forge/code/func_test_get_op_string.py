import pytest
import cirq
from cirq_pasqal import PasqalNoiseModel, PasqalDevice
from cirq.ops import NamedQubit
def test_get_op_string():
    p_qubits = cirq.NamedQubit.range(2, prefix='q')
    p_device = PasqalDevice(p_qubits)
    noise_model = PasqalNoiseModel(p_device)
    circuit = cirq.Circuit()
    circuit.append(cirq.ops.HPowGate(exponent=0.5).on(p_qubits[0]))
    with pytest.raises(ValueError, match='Got unknown operation:'):
        for moment in circuit._moments:
            _ = noise_model.noisy_moment(moment, p_qubits)
    with pytest.raises(ValueError, match='Got unknown operation:'):
        _ = noise_model.get_op_string(circuit)