import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def ssa_greedy_optimize(inputs, output, sizes, choose_fn=None, cost_fn='memory-removed'):
    """
    This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        return [(0,)]
    cost_fn = _COST_FNS.get(cost_fn, cost_fn)
    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        push_all = True
    inputs = list(map(frozenset, inputs))
    output = frozenset(output) | frozenset.intersection(*inputs)
    remaining = {}
    ssa_ids = itertools.count(len(inputs))
    ssa_path = []
    for ssa_id, key in enumerate(inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)
    dim_ref_counts = {count: set((dim for dim, keys in dim_to_keys.items() if len(keys) >= count)) - output for count in [2, 3]}
    footprints = {key: helpers.compute_size_by_dict(key, sizes) for key in remaining}
    queue = []
    for dim, keys in dim_to_keys.items():
        keys = sorted(keys, key=remaining.__getitem__)
        for i, k1 in enumerate(keys[:-1]):
            k2s = keys[1 + i:]
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn)
    while queue:
        con = choose_fn(queue, remaining)
        if con is None:
            continue
        cost, k1, k2, k12 = con
        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)
        footprints[k12] = helpers.compute_size_by_dict(k12, sizes)
        k1 = k12
        k2s = set((k2 for dim in k1 for k2 in dim_to_keys[dim]))
        k2s.discard(k1)
        if k2s:
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn)
    queue = [(helpers.compute_size_by_dict(key & output, sizes), ssa_id, key) for key, ssa_id in remaining.items()]
    heapq.heapify(queue)
    _, ssa_id1, k1 = heapq.heappop(queue)
    while queue:
        _, ssa_id2, k2 = heapq.heappop(queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = helpers.compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(queue, (cost, ssa_id12, k12))
    return ssa_path