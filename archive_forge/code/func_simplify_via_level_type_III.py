from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def simplify_via_level_type_III(link, max_consecutive_failures=100):
    """
    Applies a series of type III moves to the link, simplifying it via type
    I and II moves whenever possible.
    """
    failures, success = (0, False)
    if basic_simplify(link):
        success = True
    while failures < max_consecutive_failures:
        poss_moves = possible_type_III_moves(link)
        if len(poss_moves) == 0:
            break
        reidemeister_III(link, random.choice(poss_moves))
        if basic_simplify(link):
            failures = 0
            success = True
        else:
            failures += 1
    link._build_components()
    return success