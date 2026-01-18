import pytest
import cirq
import networkx as nx
def test_griddevice_metadata_equality():
    qubits = cirq.GridQubit.rect(2, 3)
    qubit_pairs = [(a, b) for a in qubits for b in qubits if a != b and a.is_adjacent(b)]
    gateset = cirq.Gateset(cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZ, cirq.SQRT_ISWAP)
    duration = {cirq.GateFamily(cirq.XPowGate): cirq.Duration(nanos=1), cirq.GateFamily(cirq.YPowGate): cirq.Duration(picos=3), cirq.GateFamily(cirq.ZPowGate): cirq.Duration(picos=2), cirq.GateFamily(cirq.CZ): cirq.Duration(nanos=4), cirq.GateFamily(cirq.SQRT_ISWAP): cirq.Duration(nanos=5)}
    duration2 = {cirq.GateFamily(cirq.XPowGate): cirq.Duration(nanos=10), cirq.GateFamily(cirq.YPowGate): cirq.Duration(picos=13), cirq.GateFamily(cirq.ZPowGate): cirq.Duration(picos=12), cirq.GateFamily(cirq.CZ): cirq.Duration(nanos=14), cirq.GateFamily(cirq.SQRT_ISWAP): cirq.Duration(nanos=15)}
    isolated_qubits = [cirq.GridQubit(9, 9)]
    target_gatesets = [cirq.CZTargetGateset(), cirq.SqrtIswapTargetGateset()]
    metadata = cirq.GridDeviceMetadata(qubit_pairs, gateset, gate_durations=duration)
    metadata2 = cirq.GridDeviceMetadata(qubit_pairs[:2], gateset, gate_durations=duration)
    metadata3 = cirq.GridDeviceMetadata(qubit_pairs, gateset, gate_durations=None)
    metadata4 = cirq.GridDeviceMetadata(qubit_pairs, gateset, gate_durations=duration2)
    metadata5 = cirq.GridDeviceMetadata(reversed(qubit_pairs), gateset, gate_durations=duration)
    metadata6 = cirq.GridDeviceMetadata(qubit_pairs, gateset, gate_durations=duration, all_qubits=qubits + isolated_qubits)
    metadata7 = cirq.GridDeviceMetadata(qubit_pairs, gateset, compilation_target_gatesets=target_gatesets)
    metadata8 = cirq.GridDeviceMetadata(qubit_pairs, gateset, compilation_target_gatesets=target_gatesets[::-1])
    metadata9 = cirq.GridDeviceMetadata(qubit_pairs, gateset, compilation_target_gatesets=tuple(target_gatesets))
    metadata10 = cirq.GridDeviceMetadata(qubit_pairs, gateset, compilation_target_gatesets=set(target_gatesets))
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(metadata)
    eq.add_equality_group(metadata2)
    eq.add_equality_group(metadata3)
    eq.add_equality_group(metadata4)
    eq.add_equality_group(metadata6)
    eq.add_equality_group(metadata7, metadata8, metadata9, metadata10)
    assert metadata == metadata5