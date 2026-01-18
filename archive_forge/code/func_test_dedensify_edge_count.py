import pytest
import networkx as nx
def test_dedensify_edge_count(self):
    """
        Verifies that dedensify produced the correct number of edges in an
        undirected graph
        """
    G = self.build_original_graph()
    c_G, c_nodes = nx.dedensify(G, threshold=2, copy=True)
    compressed_edge_count = len(c_G.edges())
    verified_original_edge_count = len(G.edges())
    assert compressed_edge_count <= verified_original_edge_count
    verified_compressed_G = self.build_compressed_graph()
    verified_compressed_edge_count = len(verified_compressed_G.edges())
    assert compressed_edge_count == verified_compressed_edge_count