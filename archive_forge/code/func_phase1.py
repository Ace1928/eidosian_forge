import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
def phase1(self):
    rem_deg = self.remaining_degree
    while sum(rem_deg.values()) >= 2 * self.dmax ** 2:
        u, v = sorted(random_weighted_sample(rem_deg, 2, self.rng))
        if self.graph.has_edge(u, v):
            continue
        if self.rng.random() < self.p(u, v):
            self.graph.add_edge(u, v)
            self.update_remaining(u, v)