from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def pickup_simplify(link, type_III=0):
    """
    Optimizes the overcrossings on a diagram, then the undercrossings,
    simplifying in between until the process stabilizes.
    """
    L = link
    init_num_crossings = len(L.crossings)
    if type_III:
        simplify_via_level_type_III(link, type_III)
    else:
        basic_simplify(L, build_components=False)
    stabilized = init_num_crossings == 0
    while not stabilized:
        old_cross = len(L.crossings)
        strand_pickup(L, 'over')
        if type_III:
            simplify_via_level_type_III(link, type_III)
        strand_pickup(L, 'under')
        if type_III:
            simplify_via_level_type_III(link, type_III)
        new_cross = len(L.crossings)
        stabilized = new_cross == 0 or new_cross == old_cross
    L._rebuild()
    return len(L.crossings) != init_num_crossings