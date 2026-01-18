import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_greedy_suboptimal_branching1a():
    G = build_branching(greedy_subopt_branching_1a)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 128