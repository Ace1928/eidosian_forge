import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('node_shape', ('o', 's'))
def test_draw_edges_min_source_target_margins(node_shape):
    """Test that there is a wider gap between the node and the start of an
    incident edge when min_source_margin is specified.

    This test checks that the use of min_{source/target}_margin kwargs result
    in shorter (more padding) between the edges and source and target nodes.
    As a crude visual example, let 's' and 't' represent source and target
    nodes, respectively:

       Default:
       s-----------------------------t

       With margins:
       s   -----------------------   t

    """
    fig, ax = plt.subplots()
    G = nx.DiGraph([(0, 1)])
    pos = {0: (0, 0), 1: (1, 0)}
    default_patch = nx.draw_networkx_edges(G, pos, ax=ax, node_shape=node_shape)[0]
    default_extent = default_patch.get_extents().corners()[::2, 0]
    padded_patch = nx.draw_networkx_edges(G, pos, ax=ax, node_shape=node_shape, min_source_margin=100, min_target_margin=100)[0]
    padded_extent = padded_patch.get_extents().corners()[::2, 0]
    assert padded_extent[0] > default_extent[0]
    assert padded_extent[1] < default_extent[1]