from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def match_known_group(G, alt=None):
    needed = [g.order() for g in G.generators]
    return search([], needed, G.order(), alt=alt, profile=order_profile(G))