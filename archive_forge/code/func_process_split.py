import sys
from sympy.external.importtools import version_tuple
import pytest
from sympy.core.cache import clear_cache, USE_CACHE
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from sympy.utilities.misc import ARCH
import re
def process_split(config, items):
    split = config.getoption('--split')
    if not split:
        return
    m = sp.match(split)
    if not m:
        raise ValueError('split must be a string of the form a/b where a and b are ints.')
    i, t = map(int, m.groups())
    start, end = ((i - 1) * len(items) // t, i * len(items) // t)
    if i < t:
        del items[end:]
    del items[:start]