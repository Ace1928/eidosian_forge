from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def isotope_to_braid(link):
    """
    Performs Reidemester II moves until the Seifert tree becomes a chain, i.e.
    the Seifert circles are all nested and compatibly oriented, following
    P. Vogel, "Representation of links by braids, a new algorithm"
    """
    while remove_admissible_move(link):
        pass