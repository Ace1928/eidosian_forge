import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_non_repeated_cuts():
    K = nx.karate_club_graph()
    bcc = max(list(nx.biconnected_components(K)), key=len)
    G = K.subgraph(bcc)
    solution = [{32, 33}, {2, 33}, {0, 3}, {0, 1}, {29, 33}]
    cuts = list(nx.all_node_cuts(G))
    if len(solution) != len(cuts):
        print(f'Solution: {solution}')
        print(f'Result: {cuts}')
    assert len(solution) == len(cuts)
    for cut in cuts:
        assert cut in solution