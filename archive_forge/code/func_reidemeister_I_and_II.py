from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def reidemeister_I_and_II(link, A):
    """
    Does a type-1 or type-2 simplification at the given crossing A if
    possible.

    Returns the pair: {crossings eliminated}, {crossings changed}
    """
    eliminated, changed = reidemeister_I(link, A)
    if not eliminated:
        for a in range(4):
            (B, b), (C, c) = (A.adjacent[a], A.adjacent[a + 1])
            if B == C and (b - 1) % 4 == c and ((a + b) % 2 == 0):
                eliminated, changed = reidemeister_I(link, B)
                if eliminated:
                    break
                else:
                    W, w = A.adjacent[a + 2]
                    X, x = A.adjacent[a + 3]
                    Y, y = B.adjacent[b + 1]
                    Z, z = B.adjacent[b + 2]
                    eliminated = set([A, B])
                    if W != B:
                        W[w] = Z[z]
                        changed.update(set([W, Z]))
                    if X != B:
                        X[x] = Y[y]
                        changed.update(set([X, Y]))
                    remove_crossings(link, eliminated)
                    break
    return (eliminated, changed)