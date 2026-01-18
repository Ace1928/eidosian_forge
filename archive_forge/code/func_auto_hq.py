import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def auto_hq(inputs, output, size_dict, memory_limit=None):
    """Finds the contraction path by automatically choosing the method based on
    how many input arguments there are, but targeting a more generous
    amount of search time than ``'auto'``.
    """
    from .path_random import random_greedy_128
    N = len(inputs)
    return _AUTO_HQ_CHOICES.get(N, random_greedy_128)(inputs, output, size_dict, memory_limit)