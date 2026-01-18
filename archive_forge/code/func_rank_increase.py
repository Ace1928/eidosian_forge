import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def rank_increase(a, b):
    if len(a) == 0:
        return True
    elif len(a) == 1:
        return a[0] != b
    elif len(a) == 4:
        return False
    l = a + [b]
    w = cross_product(subtract(l[1], l[0]), subtract(l[2], l[0]))
    if len(a) == 2:
        return any(w)
    elif len(a) == 3:
        return dot_product(w, subtract(l[3], l[0])) != 0
    else:
        raise Exception("This shouldn't be possible.")