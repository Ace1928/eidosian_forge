import itertools
import os
import re
import numpy as np
def sublists(lst, min_elmts=0, max_elmts=None):
    """Build a list of all possible sublists of a given list. Restrictions
    on the length of the sublists can be posed via the min_elmts and max_elmts
    parameters.
    All sublists
    have will have at least min_elmts elements and not more than max_elmts
    elements.

    Parameters
    ----------
    lst : list
        Original list from which sublists are generated.
    min_elmts : int
        Lower bound for the length of sublists.
    max_elmts : int or None
        If int, then max_elmts are the upper bound for the length of sublists.
        If None, sublists' length is not restricted. In this case the longest
        sublist will be of the same length as the original list lst.

    Returns
    -------
    result : list
        A list of all sublists of lst fulfilling the length restrictions.
    """
    if max_elmts is None:
        max_elmts = len(lst)
    result = itertools.chain.from_iterable((itertools.combinations(lst, sublist_len) for sublist_len in range(min_elmts, max_elmts + 1)))
    if type(result) is not list:
        result = list(result)
    return result