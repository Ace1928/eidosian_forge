from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
def find_best_reset_points(activation_sizes: List[int], num_checkpoints: int) -> Tuple[int, List[int]]:
    """
    Assuming constant memory requirement from the model, its gradients
    and the associated optimizer state (realistic for small models
    or models that are sharded enough to be considered small), this
    function computes the ideal placement for the checkpoints by
    returning the limits at which we should reset memory.
    """
    n = len(activation_sizes)

    @lru_cache(maxsize=None)
    def visit(pos: int, remaining: int) -> Tuple[int, List[int]]:
        if pos == n:
            return (0, [])
        if remaining == 0:
            return (sum(activation_sizes[pos:]), [])
        min_val = float('inf')
        allocation = []
        current_chunk = 0
        for curr_pos in range(pos, n):
            current_chunk += activation_sizes[curr_pos]
            sub_result, sub_alloc = visit(curr_pos + 1, remaining - 1)
            result = max(current_chunk, sub_result)
            if result < min_val:
                min_val = result
                allocation = list(sub_alloc)
                allocation.append(curr_pos + 1)
        return (int(min_val), allocation)
    best_score, best_allocation = visit(0, num_checkpoints)
    return (best_score, best_allocation[::-1])