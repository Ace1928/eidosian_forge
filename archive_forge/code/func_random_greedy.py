import functools
import heapq
import math
import numbers
import time
from collections import deque
from . import helpers, paths
def random_greedy(inputs, output, idx_dict, memory_limit=None, **optimizer_kwargs):
    """
    """
    optimizer = RandomGreedy(**optimizer_kwargs)
    return optimizer(inputs, output, idx_dict, memory_limit)