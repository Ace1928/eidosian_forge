import pytest
import networkx as nx
def test_circular_planar_and_shell_dim_error(self):
    G = nx.path_graph(4)
    pytest.raises(ValueError, nx.circular_layout, G, dim=1)
    pytest.raises(ValueError, nx.shell_layout, G, dim=1)
    pytest.raises(ValueError, nx.shell_layout, G, dim=3)
    pytest.raises(ValueError, nx.planar_layout, G, dim=1)
    pytest.raises(ValueError, nx.planar_layout, G, dim=3)