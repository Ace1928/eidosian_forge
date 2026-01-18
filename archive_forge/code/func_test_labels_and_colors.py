import itertools
import os
import warnings
import pytest
import networkx as nx
def test_labels_and_colors():
    G = nx.cubical_graph()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2, 3], node_color='r', node_size=500, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color='b', node_size=500, alpha=[0.25, 0.5, 0.75, 1.0])
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)], width=8, alpha=0.5, edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)], width=8, alpha=0.5, edge_color='b')
    nx.draw_networkx_edges(G, pos, edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)], arrows=True, min_source_margin=0.5, min_target_margin=0.75, width=8, edge_color='b')
    labels = {}
    labels[0] = '$a$'
    labels[1] = '$b$'
    labels[2] = '$c$'
    labels[3] = '$d$'
    labels[4] = '$\\alpha$'
    labels[5] = '$\\beta$'
    labels[6] = '$\\gamma$'
    labels[7] = '$\\delta$'
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=None, rotate=False)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(4, 5): '4-5'})