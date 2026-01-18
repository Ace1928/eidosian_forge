import itertools
import os
import warnings
import pytest
import networkx as nx
def test_multigraph_edgelist_tuples():
    G = nx.path_graph(3, create_using=nx.MultiDiGraph)
    nx.draw_networkx(G, edgelist=[(0, 1, 0)])
    nx.draw_networkx(G, edgelist=[(0, 1, 0)], node_size=[10, 20, 0])