from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_gh6787_variable_weighted_complete_graph(self):
    N = 8
    cg = nx.complete_graph(N)
    cg.add_weighted_edges_from([(u, v, 9) for u, v in cg.edges])
    cg.add_weighted_edges_from([(u, v, 1) for u, v in nx.cycle_graph(N).edges])
    mcb = nx.minimum_cycle_basis(cg, weight='weight')
    check_independent(mcb)