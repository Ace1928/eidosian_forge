import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
def p(self, u, v):
    return 1 - self.degree[u] * self.degree[v] / (4.0 * self.m)