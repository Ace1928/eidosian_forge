import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def shallow_copy_node_attr(self, H, G):
    assert G.nodes[0]['foo'] == H.nodes[0]['foo']
    G.nodes[0]['foo'].append(1)
    assert G.nodes[0]['foo'] == H.nodes[0]['foo']