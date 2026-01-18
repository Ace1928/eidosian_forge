import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
def create_decorator(_f, with_self):
    if with_self:

        def f(self, *args, **kwargs):
            return _f(self, *args, **kwargs)
    else:

        def f(self, *args, **kwargs):
            return _f(*args, **kwargs)
    return f