from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def finish_up(name, G):
    found[name] = G
    if print_report:
        print('=' * 40)
        print(f'{name}:')
        print(G.generators)