import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_empty_compose_all():
    with pytest.raises(ValueError):
        nx.compose_all([])