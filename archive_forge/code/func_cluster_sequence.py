from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def cluster_sequence(creation_sequence):
    """
    Return cluster sequence for the given threshold graph creation sequence.
    """
    triseq = triangle_sequence(creation_sequence)
    degseq = degree_sequence(creation_sequence)
    cseq = []
    for i, deg in enumerate(degseq):
        tri = triseq[i]
        if deg <= 1:
            cseq.append(0)
            continue
        max_size = deg * (deg - 1) // 2
        cseq.append(tri / max_size)
    return cseq