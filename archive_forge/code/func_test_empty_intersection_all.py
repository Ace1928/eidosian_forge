import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_empty_intersection_all():
    with pytest.raises(ValueError):
        nx.intersection_all([])