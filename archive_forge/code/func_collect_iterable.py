from __future__ import print_function, unicode_literals
import itertools
from collections import OrderedDict, deque
from functools import wraps
from types import GeneratorType
from six.moves import zip_longest
from .py3compat import fix_unicode_literals_in_doctest
def collect_iterable(f):

    @wraps(f)
    def new_f(*args, **kwargs):
        return list(f(*args, **kwargs))
    return new_f