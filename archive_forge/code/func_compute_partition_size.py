import math
import fileio
import collections
import platform
import multiprocessing as multiproc
import random
from functools import reduce
from itertools import chain, count, islice, takewhile
from typing import List, Optional, Dict
def compute_partition_size(result, processes):
    """
    Attempts to compute the partition size to evenly distribute work across processes. Defaults to
    1 if the length of result cannot be determined.
    :param result: Result to compute on
    :param processes: Number of processes to use
    :return: Best partition size
    """
    try:
        return max(math.ceil(len(result) / processes), 1)
    except TypeError:
        return 1