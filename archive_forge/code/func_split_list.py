import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def split_list(l, split, density=None):
    """
    Splits a list into part a of b

    split should be a string of the form 'a/b'. For instance, '1/3' would give
    the split one of three.

    If the length of the list is not divisible by the number of splits, the
    last split will have more items.

    `density` may be specified as a list.  If specified,
    tests will be balanced so that each split has as equal-as-possible
    amount of mass according to `density`.

    >>> from sympy.testing.runtests import split_list
    >>> a = list(range(10))
    >>> split_list(a, '1/3')
    [0, 1, 2]
    >>> split_list(a, '2/3')
    [3, 4, 5]
    >>> split_list(a, '3/3')
    [6, 7, 8, 9]
    """
    m = sp.match(split)
    if not m:
        raise ValueError('split must be a string of the form a/b where a and b are ints')
    i, t = map(int, m.groups())
    if not density:
        return l[(i - 1) * len(l) // t:i * len(l) // t]
    tot = sum(density)
    density = [x / tot for x in density]

    def density_inv(x):
        """Interpolate the inverse to the cumulative
        distribution function given by density"""
        if x <= 0:
            return 0
        if x >= sum(density):
            return 1
        cumm = 0
        for i, d in enumerate(density):
            cumm += d
            if cumm >= x:
                break
        frac = (d - (cumm - x)) / d
        return (i + frac) / len(density)
    lower_frac = density_inv((i - 1) / t)
    higher_frac = density_inv(i / t)
    return l[int(lower_frac * len(l)):int(higher_frac * len(l))]