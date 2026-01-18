from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture(name='asia_graph')
def asia_graph_fixture():
    return asia_graph()