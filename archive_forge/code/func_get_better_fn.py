import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def get_better_fn(key):
    return _BETTER_FNS[key]