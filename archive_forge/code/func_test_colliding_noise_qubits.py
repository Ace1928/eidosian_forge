import cirq
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG, OpIdentifier
def test_colliding_noise_qubits():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    op_id0 = OpIdentifier(cirq.CZPowGate)
    model = InsertionNoiseModel({op_id0: cirq.CNOT(q1, q2)}, require_physical_tag=False)
    moment_0 = cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1, q2, q3]) == [moment_0, cirq.Moment(cirq.CNOT(q1, q2)), cirq.Moment(cirq.CNOT(q1, q2))]
    cirq.testing.assert_equivalent_repr(model)