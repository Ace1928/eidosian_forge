import pytest
import networkx as nx
import cirq
def test_namedqubit_device():
    nx_graph = nx.Graph([('a', 'b'), ('a', 'c'), ('a', 'd')])
    device = cirq.testing.RoutingTestingDevice(nx_graph)
    relabeled_graph = device.metadata.nx_graph
    qubit_set = {cirq.NamedQubit(n) for n in 'abcd'}
    assert set(relabeled_graph.nodes) == qubit_set
    assert nx.is_isomorphic(nx_graph, relabeled_graph)