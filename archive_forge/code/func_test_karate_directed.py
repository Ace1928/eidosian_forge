import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_karate_directed():
    G = nx.karate_club_graph().to_directed()
    _check_edge_connectivity(G)