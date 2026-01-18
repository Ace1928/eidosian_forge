import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def fix_point(fn, dct):
    """Helper function to run fix-point algorithm.
        """
    old_point = None
    new_point = fix_point_progress(dct)
    while old_point != new_point:
        fn(dct)
        old_point = new_point
        new_point = fix_point_progress(dct)