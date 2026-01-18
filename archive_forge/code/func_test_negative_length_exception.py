import pytest
import networkx as nx
def test_negative_length_exception():
    G = nx.cycle_graph(3)
    with pytest.raises(ValueError):
        nx.number_of_walks(G, -1)