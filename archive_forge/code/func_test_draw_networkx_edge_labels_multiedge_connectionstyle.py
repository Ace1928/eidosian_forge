import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize(('G', 'expected_n_edges'), ([nx.DiGraph(), 2], [nx.MultiGraph(), 4], [nx.MultiDiGraph(), 4]))
def test_draw_networkx_edge_labels_multiedge_connectionstyle(G, expected_n_edges):
    """Draws labels correctly for 3 types of graphs and checks for valid length and class names"""
    for i, (u, v) in enumerate([(0, 1), (0, 1), (0, 1), (0, 2)]):
        G.add_edge(u, v, weight=round(i / 3, 2))
    pos = {n: (n, n) for n in G}
    arrows = nx.draw_networkx_edges(G, pos, connectionstyle=['arc3,rad=0.1', 'arc3,rad=0.1', 'arc3,rad=0.1'])
    for conn_style in ['arc3,rad=0.1', ['arc3,rad=0.1', 'arc3,rad=0.2'], ['arc3,rad=0.1', 'arc3,rad=0.1', 'arc3,rad=0.1']]:
        text_items = nx.draw_networkx_edge_labels(G, pos, connectionstyle=conn_style)
        assert len(text_items) == expected_n_edges
        for ti in text_items.values():
            assert ti.__class__.__name__ == 'CurvedArrowText'