import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.mpl_image_compare
def test_house_with_colors():
    G = nx.house_graph()
    fig, ax = plt.subplots()
    pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}
    nx.draw_networkx_nodes(G, pos, node_size=3000, nodelist=[0, 1, 2, 3], node_color='tab:blue')
    nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[4], node_color='tab:orange')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
    ax.margins(0.11)
    plt.tight_layout()
    plt.axis('off')
    return fig