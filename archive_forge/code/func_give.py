import ``rng`` and access the method directly. For example, to capture the
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
import random as _random
def give(a, b, seq=seed):
    a, b = (as_int(a), as_int(b))
    w = b - a
    if w < 0:
        raise ValueError('_randint got empty range')
    try:
        x = seq.pop()
    except IndexError:
        raise ValueError('_randint sequence was too short')
    if a <= x <= b:
        return x
    else:
        return give(a, b, seq)