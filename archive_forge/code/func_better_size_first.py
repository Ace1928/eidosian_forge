import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def better_size_first(flops, size, best_flops, best_size):
    return (size, flops) < (best_size, best_flops)