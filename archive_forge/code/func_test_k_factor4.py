import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_k_factor4(self):
    g = gen.lattice.hexagonal_lattice_graph(4, 4)
    with pytest.raises(nx.NetworkXUnfeasible):
        reg.k_factor(g, 2)