import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize(('G', 'expected_n_edges'), ([nx.DiGraph(), 2], [nx.MultiGraph(), 4], [nx.MultiDiGraph(), 4]))
def test_draw_networkx_edges_multiedge_connectionstyle(G, expected_n_edges):
    """Draws edges correctly for 3 types of graphs and checks for valid length"""
    for i, (u, v) in enumerate([(0, 1), (0, 1), (0, 1), (0, 2)]):
        G.add_edge(u, v, weight=round(i / 3, 2))
    pos = {n: (n, n) for n in G}
    for conn_style in ['arc3,rad=0.1', ['arc3,rad=0.1', 'arc3,rad=0.1'], ['arc3,rad=0.1', 'arc3,rad=0.1', 'arc3,rad=0.2']]:
        nx.draw_networkx_edges(G, pos, connectionstyle=conn_style)
        arrows = nx.draw_networkx_edges(G, pos, connectionstyle=conn_style)
        assert len(arrows) == expected_n_edges