from subprocess import check_output
import os.path
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target
from all overloads.
def format_signature(sig):

    def fmt(c):
        try:
            return c.__name__
        except AttributeError:
            return repr(c).strip('\'"')
    out = tuple(map(fmt, sig))
    return '`({0})`'.format(', '.join(out))