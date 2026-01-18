import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_digraphs(self):
    for dest, source in [(to_dict_of_dicts, from_dict_of_dicts), (to_dict_of_lists, from_dict_of_lists)]:
        G = cycle_graph(10)
        dod = dest(G)
        GG = source(dod)
        assert nodes_equal(sorted(G.nodes()), sorted(GG.nodes()))
        assert edges_equal(sorted(G.edges()), sorted(GG.edges()))
        GW = to_networkx_graph(dod)
        assert nodes_equal(sorted(G.nodes()), sorted(GW.nodes()))
        assert edges_equal(sorted(G.edges()), sorted(GW.edges()))
        GI = nx.Graph(dod)
        assert nodes_equal(sorted(G.nodes()), sorted(GI.nodes()))
        assert edges_equal(sorted(G.edges()), sorted(GI.edges()))
        G = cycle_graph(10, create_using=nx.DiGraph)
        dod = dest(G)
        GG = source(dod, create_using=nx.DiGraph)
        assert sorted(G.nodes()) == sorted(GG.nodes())
        assert sorted(G.edges()) == sorted(GG.edges())
        GW = to_networkx_graph(dod, create_using=nx.DiGraph)
        assert sorted(G.nodes()) == sorted(GW.nodes())
        assert sorted(G.edges()) == sorted(GW.edges())
        GI = nx.DiGraph(dod)
        assert sorted(G.nodes()) == sorted(GI.nodes())
        assert sorted(G.edges()) == sorted(GI.edges())