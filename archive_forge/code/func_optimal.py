import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def optimal(inputs, output, size_dict, memory_limit=None):
    """
    Computes all possible pair contractions in a depth-first recursive manner,
    sieving results based on ``memory_limit`` and the best path found so far.
    Returns the lowest cost path. This algorithm scales factoriallly with
    respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    inputs : list
        List of sets that represent the lhs side of the einsum subscript.
    output : set
        Set that represents the rhs side of the overall einsum subscript.
    size_dict : dictionary
        Dictionary of index sizes.
    memory_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> optimal(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """
    inputs = tuple(map(frozenset, inputs))
    output = frozenset(output)
    best = {'flops': float('inf'), 'ssa_path': (tuple(range(len(inputs))),)}
    size_cache = {}
    result_cache = {}

    def _optimal_iterate(path, remaining, inputs, flops):
        if len(remaining) == 1:
            best['flops'] = flops
            best['ssa_path'] = path
            return
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = (j, i)
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)
            new_flops = flops + flops12
            if new_flops >= best['flops']:
                continue
            if memory_limit not in _UNLIMITED_MEM:
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)
                if size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                    if new_flops < best['flops']:
                        best['flops'] = new_flops
                        best['ssa_path'] = path + (tuple(remaining),)
                    continue
            _optimal_iterate(path=path + ((i, j),), inputs=inputs + (k12,), remaining=remaining - {i, j} | {len(inputs)}, flops=new_flops)
    _optimal_iterate(path=(), inputs=inputs, remaining=set(range(len(inputs))), flops=0)
    return ssa_to_linear(best['ssa_path'])