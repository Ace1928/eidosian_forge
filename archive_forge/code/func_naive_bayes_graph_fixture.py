from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture(name='naive_bayes_graph')
def naive_bayes_graph_fixture():
    return naive_bayes_graph()