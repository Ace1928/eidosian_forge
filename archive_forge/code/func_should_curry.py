import toolz
import toolz.curried
from toolz.curried import (take, first, second, sorted, merge_with, reduce,
from collections import defaultdict
from importlib import import_module
from operator import add
def should_curry(func):
    if not callable(func) or isinstance(func, toolz.curry):
        return False
    nargs = toolz.functoolz.num_required_args(func)
    if nargs is None or nargs > 1:
        return True
    return nargs == 1 and toolz.functoolz.has_keywords(func)