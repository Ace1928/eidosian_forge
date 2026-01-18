import pytest
import cirq
import cirq_google
@pytest.mark.parametrize('before, gate_family', [(cirq.Circuit(cirq.Z(_qa) ** 0.5, cirq.CZ(_qa, _qb)), cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()])), (cirq.Circuit((cirq.Z ** 0.5)(_qa).with_tags(cirq_google.PhysicalZTag()), cirq.CZ(_qa, _qb)), cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()])), (cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa), cirq.CZ(_qa, _qb)), cirq.GateFamily(cirq.PhasedXPowGate))])
def test_eject_paulis_disabled(before, gate_family):
    after = cirq.optimize_for_target_gateset(before, gateset=cirq_google.GoogleCZTargetGateset(additional_gates=[gate_family]), ignore_failures=False)
    cirq.testing.assert_same_circuits(after, before)