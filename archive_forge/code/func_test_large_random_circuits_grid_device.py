import networkx as nx
import pytest
import cirq
@pytest.mark.parametrize('qubits, n_moments, op_density, random_state', [(30, size, 0.5, seed) for size in [50, 100] for seed in range(2)])
def test_large_random_circuits_grid_device(qubits: int, n_moments: int, op_density: float, random_state: int):
    c_orig = cirq.testing.random_circuit(qubits=qubits, n_moments=n_moments, op_density=op_density, random_state=random_state)
    mapping = glob_mapper.initial_mapping(c_orig)
    assert len(set(mapping.values())) == len(mapping.values())
    assert set(mapping.keys()) == set(c_orig.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(glob_device_graph, mapping.values()))