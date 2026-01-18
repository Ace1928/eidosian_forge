from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def elts_by_order(G):
    """Sort the elements of a group by their order. """
    elts = defaultdict(list)
    for g in G.elements:
        elts[g.order()].append(g)
    return elts