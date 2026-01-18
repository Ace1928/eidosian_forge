import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def better_flops_first(flops, size, best_flops, best_size):
    return (flops, size) < (best_flops, best_size)