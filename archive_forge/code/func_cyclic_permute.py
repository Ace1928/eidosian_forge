from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def cyclic_permute(l, n):
    L = len(l)
    return [l[(i + n) % L] for i in range(L)]