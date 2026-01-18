from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def random_reverse_move(link, t, n):
    """
    Performs a crossing increasing move of type t, where t is 1, 2, or 3
    n is for labeling the new crossings
    """
    if t == 1:
        random_reverse_type_I(link, 'new' + str(n))
    elif t == 2:
        random_reverse_type_II(link, 'new' + str(n), 'new' + str(n + 1))
    else:
        poss_moves = possible_type_III_moves(link)
        if len(poss_moves) != 0:
            reidemeister_III(link, random.choice(poss_moves))