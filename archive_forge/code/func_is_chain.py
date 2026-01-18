from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def is_chain(tree):
    tails = [e[0] for e in tree]
    heads = [e[1] for e in tree]
    return len(set(tails)) == len(tails) and len(set(heads)) == len(heads)