import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def get_dimensionality_histogram(ranks, roots):
    h = [0, 0, 0, 0]
    for e in roots:
        h[ranks[e]] += 1
    return tuple(h)