from itertools import combinations
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def minimum_edge_weight(self, T, u, v):
    path = nx.shortest_path(T, u, v, weight='weight')
    return min(((T[u][v]['weight'], (u, v)) for u, v in zip(path, path[1:])))