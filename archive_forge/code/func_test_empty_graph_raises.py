import pytest
import networkx as nx
from networkx import is_strongly_regular
@pytest.mark.parametrize('f', (nx.is_distance_regular, nx.is_strongly_regular))
def test_empty_graph_raises(f):
    G = nx.Graph()
    with pytest.raises(nx.NetworkXPointlessConcept, match='Graph has no nodes'):
        f(G)