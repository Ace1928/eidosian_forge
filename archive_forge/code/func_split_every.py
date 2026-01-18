import math
import fileio
import collections
import platform
import multiprocessing as multiproc
import random
from functools import reduce
from itertools import chain, count, islice, takewhile
from typing import List, Optional, Dict
def split_every(parts, iterable):
    """
    Split an iterable into parts of length parts
    :param iterable: iterable to split
    :param parts: number of chunks
    :return: return the iterable split in parts
    """
    return takewhile(bool, (list(islice(iterable, parts)) for _ in count()))