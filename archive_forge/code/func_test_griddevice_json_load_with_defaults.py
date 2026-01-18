import pytest
import cirq
import networkx as nx
def test_griddevice_json_load_with_defaults():
    qubits = cirq.GridQubit.rect(2, 3)
    qubit_pairs = [(a, b) for a in qubits for b in qubits if a != b and a.is_adjacent(b)]
    gateset = cirq.Gateset(cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZ)
    metadata = cirq.GridDeviceMetadata(qubit_pairs, gateset)
    rep_str = cirq.to_json(metadata)
    assert metadata == cirq.read_json(json_text=rep_str)