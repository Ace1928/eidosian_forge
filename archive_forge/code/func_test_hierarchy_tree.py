import pytest
import networkx as nx
def test_hierarchy_tree():
    G = nx.full_rary_tree(2, 16, create_using=nx.DiGraph())
    assert nx.flow_hierarchy(G) == 1.0