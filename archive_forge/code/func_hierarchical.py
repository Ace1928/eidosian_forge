from collections import defaultdict
from operator import itemgetter
from .dimension import Dimension
from .util import merge_dimensions
def hierarchical(keys):
    """
    Iterates over dimension values in keys, taking two sets
    of dimension values at a time to determine whether two
    consecutive dimensions have a one-to-many relationship.
    If they do a mapping between the first and second dimension
    values is returned. Returns a list of n-1 mappings, between
    consecutive dimensions.
    """
    ndims = len(keys[0])
    if ndims <= 1:
        return True
    dim_vals = list(zip(*keys))
    combinations = (zip(*dim_vals[i:i + 2]) for i in range(ndims - 1))
    hierarchies = []
    for combination in combinations:
        hierarchy = True
        store1 = defaultdict(list)
        store2 = defaultdict(list)
        for v1, v2 in combination:
            if v2 not in store2[v1]:
                store2[v1].append(v2)
            previous = store1[v2]
            if previous and previous[0] != v1:
                hierarchy = False
                break
            if v1 not in store1[v2]:
                store1[v2].append(v1)
        hierarchies.append(store2 if hierarchy else {})
    return hierarchies