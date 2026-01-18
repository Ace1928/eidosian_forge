import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def intersect(collection_of_sets):
    """
    Given an collection of sets, returns the intersection of those sets.

    Parameters
    ----------
    collection_of_sets: collections.abc.Collection[set]
        A collection of sets.

    Returns
    -------
    set
        An intersection of all sets in `collection_of_sets`. Will have the same
        type as the item initially taken from `collection_of_sets`.
    """
    collection_of_sets = list(collection_of_sets)
    first = collection_of_sets.pop()
    out = reduce(set.intersection, collection_of_sets, set(first))
    return type(first)(out)