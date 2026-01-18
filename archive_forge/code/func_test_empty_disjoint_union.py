import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_empty_disjoint_union():
    with pytest.raises(ValueError):
        nx.disjoint_union_all([])