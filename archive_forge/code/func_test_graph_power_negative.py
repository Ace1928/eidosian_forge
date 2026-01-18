import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_graph_power_negative():
    with pytest.raises(ValueError):
        nx.power(nx.Graph(), -1)