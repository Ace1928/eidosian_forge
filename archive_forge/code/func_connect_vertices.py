from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def connect_vertices(e1, v1, e2, v2):
    e1[v1] = e1[v1] | e2[v2]
    e2[v2] = e1[v1] | e2[v2]