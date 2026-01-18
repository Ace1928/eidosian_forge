from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_bad_integer_labels(self):
    with pytest.raises(KeyError):
        T = nx.Graph(nx.utils.pairwise('abc'))
        nx.to_prufer_sequence(T)