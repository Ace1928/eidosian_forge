import cirq
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG, OpIdentifier
def test_insertion_noise():
    q0, q1 = cirq.LineQubit.range(2)
    op_id0 = OpIdentifier(cirq.XPowGate, q0)
    op_id1 = OpIdentifier(cirq.ZPowGate, q1)
    model = InsertionNoiseModel({op_id0: cirq.T(q0), op_id1: cirq.H(q1)}, require_physical_tag=False)
    assert not model.prepend
    moment_0 = cirq.Moment(cirq.X(q0), cirq.X(q1))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1]) == [moment_0, cirq.Moment(cirq.T(q0))]
    moment_1 = cirq.Moment(cirq.Z(q0), cirq.Z(q1))
    assert model.noisy_moment(moment_1, system_qubits=[q0, q1]) == [moment_1, cirq.Moment(cirq.H(q1))]
    moment_2 = cirq.Moment(cirq.X(q0), cirq.Z(q1))
    assert model.noisy_moment(moment_2, system_qubits=[q0, q1]) == [moment_2, cirq.Moment(cirq.T(q0), cirq.H(q1))]
    moment_3 = cirq.Moment(cirq.Z(q0), cirq.X(q1))
    assert model.noisy_moment(moment_3, system_qubits=[q0, q1]) == [moment_3]
    cirq.testing.assert_equivalent_repr(model)