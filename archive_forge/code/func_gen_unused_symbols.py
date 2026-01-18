import itertools
from collections import OrderedDict
import numpy as np
def gen_unused_symbols(used, n):
    """Generate ``n`` symbols that are not already in ``used``.

    Examples
    --------
    >>> list(oe.parser.gen_unused_symbols("abd", 2))
    ['c', 'e']
    """
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1