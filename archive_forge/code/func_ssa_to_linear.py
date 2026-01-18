import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def ssa_to_linear(ssa_path):
    """
    Convert a path with static single assignment ids to a path with recycled
    linear ids. For example::

        >>> ssa_to_linear([(0, 3), (2, 4), (1, 5)])
        [(0, 3), (1, 2), (0, 1)]
    """
    ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)
    path = []
    for ssa_ids in ssa_path:
        path.append(tuple((int(ids[ssa_id]) for ssa_id in ssa_ids)))
        for ssa_id in ssa_ids:
            ids[ssa_id:] -= 1
    return path