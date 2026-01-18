import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def calc_k12_flops(inputs, output, remaining, i, j, size_dict):
    """
    Calculate the resulting indices and flops for a potential pairwise
    contraction - used in the recursive (optimal/branch) algorithms.

    Parameters
    ----------
    inputs : tuple[frozenset[str]]
        The indices of each tensor in this contraction, note this includes
        tensors unavaiable to contract as static single assignment is used ->
        contracted tensors are not removed from the list.
    output : frozenset[str]
        The set of output indices for the whole contraction.
    remaining : frozenset[int]
        The set of indices (corresponding to ``inputs``) of tensors still
        available to contract.
    i : int
        Index of potential tensor to contract.
    j : int
        Index of potential tensor to contract.
    size_dict dict[str, int]
        Size mapping of all the indices.

    Returns
    -------
    k12 : frozenset
        The resulting indices of the potential tensor.
    cost : int
        Estimated flop count of operation.
    """
    k1, k2 = (inputs[i], inputs[j])
    either = k1 | k2
    shared = k1 & k2
    keep = frozenset.union(output, *map(inputs.__getitem__, remaining - {i, j}))
    k12 = either & keep
    cost = helpers.flop_count(either, shared - keep, 2, size_dict)
    return (k12, cost)