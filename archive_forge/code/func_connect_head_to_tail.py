from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def connect_head_to_tail(e1, e2):
    e1[1] = e1[1] | e2[0]
    e2[0] = e1[1] | e2[0]