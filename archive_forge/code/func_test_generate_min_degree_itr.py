import pytest
import networkx as nx
def test_generate_min_degree_itr():
    with pytest.raises(nx.ExceededMaxIterations, match='Could not match average_degree'):
        nx.generators.community._generate_min_degree(2, 2, 1, 0.01, 0)