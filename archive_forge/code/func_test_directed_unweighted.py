import networkx as nx
from networkx.utils import pairwise
def test_directed_unweighted(self):
    G = nx.DiGraph(pairwise(range(6), cyclic=True))
    cells = nx.voronoi_cells(G, {0, 3})
    expected = {0: {0, 1, 2}, 3: {3, 4, 5}}
    assert expected == cells