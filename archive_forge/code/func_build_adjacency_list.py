import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def build_adjacency_list(parents, bonds):
    graph = np.unique(parents)
    adjacency = {e: set() for e in graph}
    for i, j, offset in bonds:
        component_a = parents[i]
        component_b = parents[j]
        adjacency[component_a].add((component_b, offset))
    return adjacency