import pytest
import networkx as nx
def test_no_filter(self):
    nf = nx.filters.no_filter
    assert nf()
    assert nf(1)
    assert nf(2, 1)