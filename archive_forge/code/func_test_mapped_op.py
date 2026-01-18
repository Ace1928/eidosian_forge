from networkx.utils.misc import graphs_equal
import pytest
import networkx as nx
import cirq
def test_mapped_op():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]
    assert mm.mapped_op(cirq.CNOT(q[1], q[3])).qubits == (cirq.NamedQubit('a'), cirq.NamedQubit('b'))
    assert mm.mapped_op(cirq.CNOT(q[3], q[4])).qubits == (cirq.NamedQubit('b'), cirq.NamedQubit('d'))
    mm.apply_swap(q_int[2], q_int[3])
    assert mm.mapped_op(cirq.CNOT(q[1], q[2])).qubits == (cirq.NamedQubit('a'), cirq.NamedQubit('b'))
    assert mm.mapped_op(cirq.CNOT(q[1], q[3])).qubits == (cirq.NamedQubit('a'), cirq.NamedQubit('c'))