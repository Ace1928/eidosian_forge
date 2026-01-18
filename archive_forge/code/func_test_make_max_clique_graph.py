import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
def test_make_max_clique_graph(self):
    """Tests that the maximal clique graph is the same as the bipartite
        clique graph after being projected onto the nodes representing the
        cliques.

        """
    G = self.G
    B = nx.make_clique_bipartite(G)
    H1 = nx.projected_graph(B, range(-5, 0))
    H1 = nx.relabel_nodes(H1, {-v: v - 1 for v in range(1, 6)})
    H2 = nx.make_max_clique_graph(G)
    assert H1.adj == H2.adj