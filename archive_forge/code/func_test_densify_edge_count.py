import pytest
import networkx as nx
def test_densify_edge_count(self):
    """
        Verifies that densification produces the correct number of edges in the
        original directed graph
        """
    compressed_G = self.build_compressed_graph()
    compressed_edge_count = len(compressed_G.edges())
    original_graph = self.densify(compressed_G, self.c_nodes)
    original_edge_count = len(original_graph.edges())
    assert compressed_edge_count <= original_edge_count
    G = self.build_original_graph()
    assert original_edge_count == len(G.edges())