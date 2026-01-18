import networkx as nx
import cirq
def test_metadata_json_load_logic():
    qubits = cirq.LineQubit.range(4)
    graph = nx.star_graph(3)
    metadata = cirq.DeviceMetadata(qubits, graph)
    str_rep = cirq.to_json(metadata)
    assert metadata == cirq.read_json(json_text=str_rep)