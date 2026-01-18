from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def is_threshold_sequence(degree_sequence):
    """
    Returns True if the sequence is a threshold degree sequence.

    Uses the property that a threshold graph must be constructed by
    adding either dominating or isolated nodes. Thus, it can be
    deconstructed iteratively by removing a node of degree zero or a
    node that connects to the remaining nodes.  If this deconstruction
    fails then the sequence is not a threshold sequence.
    """
    ds = degree_sequence[:]
    ds.sort()
    while ds:
        if ds[0] == 0:
            ds.pop(0)
            continue
        if ds[-1] != len(ds) - 1:
            return False
        ds.pop()
        ds = [d - 1 for d in ds]
    return True