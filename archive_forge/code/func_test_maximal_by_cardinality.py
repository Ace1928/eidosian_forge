import networkx as nx
from networkx.algorithms.approximation import (
def test_maximal_by_cardinality(self):
    """Tests that the maximal clique is computed according to maximum
        cardinality of the sets.

        For more information, see pull request #1531.

        """
    G = nx.complete_graph(5)
    G.add_edge(4, 5)
    clique = max_clique(G)
    assert len(clique) > 1
    G = nx.lollipop_graph(30, 2)
    clique = max_clique(G)
    assert len(clique) > 2