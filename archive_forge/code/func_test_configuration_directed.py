import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_configuration_directed():
    seeds = [67]
    for seed in seeds:
        deg_seq = nx.random_powerlaw_tree_sequence(20, seed=seed, tries=5000)
        G = nx.DiGraph(nx.configuration_model(deg_seq, seed=seed))
        G.remove_edges_from(nx.selfloop_edges(G))
        _check_edge_connectivity(G)