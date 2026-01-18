from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_aperiodic_empty_graph():
    G = nx.empty_graph(create_using=nx.DiGraph)
    with pytest.raises(nx.NetworkXPointlessConcept, match='Graph has no nodes.'):
        nx.is_aperiodic(G)