from networkx.utils.misc import graphs_equal
import pytest
import networkx as nx
import cirq
def construct_device_graph_and_mapping():
    device_graph = nx.Graph([(cirq.NamedQubit('a'), cirq.NamedQubit('b')), (cirq.NamedQubit('b'), cirq.NamedQubit('c')), (cirq.NamedQubit('c'), cirq.NamedQubit('d')), (cirq.NamedQubit('a'), cirq.NamedQubit('e')), (cirq.NamedQubit('e'), cirq.NamedQubit('d'))])
    q = cirq.LineQubit.range(5)
    initial_mapping = {q[1]: cirq.NamedQubit('a'), q[3]: cirq.NamedQubit('b'), q[2]: cirq.NamedQubit('c'), q[4]: cirq.NamedQubit('d')}
    return (device_graph, initial_mapping, q)