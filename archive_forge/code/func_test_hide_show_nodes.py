import pickle
import pytest
import networkx as nx
def test_hide_show_nodes(self):
    SubGraph = nx.subgraph_view
    for Graph in self.Graphs:
        G = nx.path_graph(4, Graph)
        SG = G.subgraph([2, 3])
        RG = SubGraph(G, filter_node=nx.filters.hide_nodes([0, 1]))
        assert SG.nodes == RG.nodes
        assert SG.edges == RG.edges
        SGC = SG.copy()
        RGC = RG.copy()
        assert SGC.nodes == RGC.nodes
        assert SGC.edges == RGC.edges