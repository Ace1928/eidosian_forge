import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_get_circuit_connectivity():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CZ(a, b), cirq.CZ(b, c), cirq.CZ(c, d), cirq.CZ(d, a))
    graph = ccr.get_circuit_connectivity(circuit)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    is_planar, _ = nx.check_planarity(graph)
    assert is_planar