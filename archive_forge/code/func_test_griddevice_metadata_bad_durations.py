import pytest
import cirq
import networkx as nx
def test_griddevice_metadata_bad_durations():
    qubits = tuple(cirq.GridQubit.rect(1, 2))
    gateset = cirq.Gateset(cirq.XPowGate, cirq.YPowGate)
    invalid_duration = {cirq.GateFamily(cirq.XPowGate): cirq.Duration(nanos=1), cirq.GateFamily(cirq.ZPowGate): cirq.Duration(picos=1)}
    with pytest.raises(ValueError, match='ZPowGate'):
        cirq.GridDeviceMetadata([qubits], gateset, gate_durations=invalid_duration)