import pytest
import cirq
import networkx as nx
def test_griddevice_json_load():
    qubits = cirq.GridQubit.rect(2, 3)
    qubit_pairs = [(a, b) for a in qubits for b in qubits if a != b and a.is_adjacent(b)]
    gateset = cirq.Gateset(cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZ)
    duration = {cirq.GateFamily(cirq.XPowGate): cirq.Duration(nanos=1), cirq.GateFamily(cirq.YPowGate): cirq.Duration(picos=2), cirq.GateFamily(cirq.ZPowGate): cirq.Duration(picos=3), cirq.GateFamily(cirq.CZ): cirq.Duration(nanos=4)}
    isolated_qubits = [cirq.GridQubit(9, 9), cirq.GridQubit(10, 10)]
    target_gatesets = [cirq.CZTargetGateset()]
    metadata = cirq.GridDeviceMetadata(qubit_pairs, gateset, gate_durations=duration, all_qubits=qubits + isolated_qubits, compilation_target_gatesets=target_gatesets)
    rep_str = cirq.to_json(metadata)
    assert metadata == cirq.read_json(json_text=rep_str)