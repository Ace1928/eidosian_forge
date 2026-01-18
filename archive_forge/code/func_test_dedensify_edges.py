import pytest
import networkx as nx
def test_dedensify_edges(self):
    """
        Verifies that dedensify produced correct compressor nodes and the
        correct edges to/from the compressor nodes in an undirected graph
        """
    G = self.build_original_graph()
    c_G, c_nodes = nx.dedensify(G, threshold=2)
    v_compressed_G = self.build_compressed_graph()
    for s, t in c_G.edges():
        o_s = ''.join(sorted(s))
        o_t = ''.join(sorted(t))
        has_compressed_edge = c_G.has_edge(s, t)
        verified_has_compressed_edge = v_compressed_G.has_edge(o_s, o_t)
        assert has_compressed_edge == verified_has_compressed_edge
    assert len(c_nodes) == len(self.c_nodes)