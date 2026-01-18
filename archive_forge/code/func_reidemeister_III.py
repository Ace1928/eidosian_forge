from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def reidemeister_III(link, triple):
    """
    Performs the given type III move.  Modifies the given link but doesn't
    update its lists of link components.
    """
    A, B, C = [t.crossing for t in triple]
    a, b, c = [t.strand_index for t in triple]
    old_border = [(C, c - 1), (C, c - 2), (A, a - 1), (A, a - 2), (B, b - 1), (B, b - 2)]
    border_strands = [insert_strand(*P) for P in old_border]
    new_boarder = [(A, a), (B, b + 1), (B, b), (C, c + 1), (C, c), (A, a + 1)]
    for i, (X, x) in enumerate(new_boarder):
        X[x] = border_strands[i][0]
    A[a - 1], B[b - 1], C[c - 1] = (B[b + 2], C[c + 2], A[a + 2])
    [S.fuse() for S in border_strands]