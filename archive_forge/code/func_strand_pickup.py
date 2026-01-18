from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def strand_pickup(link, kind):
    """
    Simplifies link by optimizing the path of the longest sequence of overcrossings.
    Returns a new link and the number of crossings removed.
    """
    G = None
    strands = over_or_under_strands(link, kind)
    for strand in randomize_within_lengths(strands):
        if len(strand) == 1:
            continue
        if G is None:
            G = dual_graph_as_nx(link)
        crossings_removed = pickup_strand(link, G, kind, strand)
        if crossings_removed != 0:
            return crossings_removed
    return 0