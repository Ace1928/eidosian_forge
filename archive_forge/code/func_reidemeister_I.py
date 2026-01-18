from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def reidemeister_I(link, C):
    """
    Does a type-1 simplification on the given crossing C if possible.

    Returns the pair: {crossings eliminated}, {crossings changed}
    """
    elim, changed = (set(), set())
    for i in range(4):
        if C.adjacent[i] == (C, (i + 1) % 4):
            (A, a), (B, b) = (C.adjacent[i + 2], C.adjacent[i + 3])
            elim = set([C])
            if C != A:
                A[a] = B[b]
                changed = set([A, B])
    remove_crossings(link, elim)
    return (elim, changed)