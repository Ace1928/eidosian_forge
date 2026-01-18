import pickle
import pytest
import networkx as nx
def test_dispatchable_are_functions():
    assert type(nx.pagerank) is type(nx.pagerank.orig_func)