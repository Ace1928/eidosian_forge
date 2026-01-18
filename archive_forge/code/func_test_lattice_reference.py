import pytest
import random
import networkx as nx
from networkx import lattice_reference, omega, random_reference, sigma
def test_lattice_reference():
    G = nx.connected_watts_strogatz_graph(50, 6, 1, seed=rng)
    Gl = lattice_reference(G, niter=1, seed=rng)
    L = nx.average_shortest_path_length(G)
    Ll = nx.average_shortest_path_length(Gl)
    assert Ll > L
    pytest.raises(nx.NetworkXError, lattice_reference, nx.Graph())
    pytest.raises(nx.NetworkXNotImplemented, lattice_reference, nx.DiGraph())
    H = nx.Graph(((0, 1), (2, 3)))
    Hl = lattice_reference(H, niter=1)