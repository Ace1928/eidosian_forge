import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_cycles(self):
    K_undir = nx.all_pairs_node_connectivity(self.cycle)
    for source in K_undir:
        for target, k in K_undir[source].items():
            assert k == 2
    K_dir = nx.all_pairs_node_connectivity(self.directed_cycle)
    for source in K_dir:
        for target, k in K_dir[source].items():
            assert k == 1