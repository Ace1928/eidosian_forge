from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
def genealogy(index, depth):
    if index not in self.genealogy_tree:
        return
    depth += 1
    if depth > max_depth:
        return
    parent_indices = self.genealogy_tree[index]
    gtree[index] = parent_indices
    for ind in parent_indices:
        if ind not in visited:
            genealogy(ind, depth)
        visited.add(ind)