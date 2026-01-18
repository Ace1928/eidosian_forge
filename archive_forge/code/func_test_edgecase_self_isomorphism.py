import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_edgecase_self_isomorphism(self):
    """
        This edgecase is one of the cases in which it is hard to find all
        symmetry elements.
        """
    graph = nx.Graph()
    nx.add_path(graph, range(5))
    graph.add_edges_from([(2, 5), (5, 6)])
    ismags = iso.ISMAGS(graph, graph)
    ismags_answer = list(ismags.find_isomorphisms(True))
    assert ismags_answer == [{n: n for n in graph.nodes}]
    graph = nx.relabel_nodes(graph, {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 4, 6: 5})
    ismags = iso.ISMAGS(graph, graph)
    ismags_answer = list(ismags.find_isomorphisms(True))
    assert ismags_answer == [{n: n for n in graph.nodes}]