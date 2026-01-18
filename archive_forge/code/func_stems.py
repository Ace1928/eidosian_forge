from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def stems(C, v):
    yield from (([u, v, w], F.has_edge(w, u)) for u, w in combinations(C[v], 2))