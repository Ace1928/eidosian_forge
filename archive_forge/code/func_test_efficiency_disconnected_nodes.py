import networkx as nx
def test_efficiency_disconnected_nodes(self):
    """
        When nodes are disconnected, efficiency is 0
        """
    assert nx.efficiency(self.G1, 1, 2) == 0