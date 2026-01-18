from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def not_implemented():
    G = nx.Graph([(1, 2), (2, 3)])
    list(nx.all_topological_sorts(G))