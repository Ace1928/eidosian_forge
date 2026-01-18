import pytest
import networkx as nx
import cirq
def test_allowed_multi_qubit_gates():
    device = cirq.testing.construct_ring_device(5)
    device.validate_operation(cirq.MeasurementGate(1).on(cirq.LineQubit(0)))
    device.validate_operation(cirq.MeasurementGate(2).on(*cirq.LineQubit.range(2)))
    device.validate_operation(cirq.MeasurementGate(3).on(*cirq.LineQubit.range(3)))
    with pytest.raises(ValueError, match='Unsupported operation'):
        device.validate_operation(cirq.CCNOT(*cirq.LineQubit.range(3)))
    device.validate_operation(cirq.CNOT(*cirq.LineQubit.range(2)))