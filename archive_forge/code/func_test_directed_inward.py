import networkx as nx
from networkx.utils import pairwise
def test_directed_inward(self):
    """Tests that reversing the graph gives the "inward" Voronoi
        partition.

        """
    G = nx.DiGraph(pairwise(range(6), cyclic=True))
    G = G.reverse(copy=False)
    cells = nx.voronoi_cells(G, {0, 3})
    expected = {0: {0, 4, 5}, 3: {1, 2, 3}}
    assert expected == cells