import math
import fileio
import collections
import platform
import multiprocessing as multiproc
import random
from functools import reduce
from itertools import chain, count, islice, takewhile
from typing import List, Optional, Dict
def is_tabulatable(val):
    if is_primitive(val):
        return False
    if is_iterable(val) or is_namedtuple(val) or isinstance(val, list):
        return True
    return False