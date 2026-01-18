import toolz
import toolz.curried
from toolz.curried import (take, first, second, sorted, merge_with, reduce,
from collections import defaultdict
from importlib import import_module
from operator import add
def curry_namespace(ns):
    return {name: toolz.curry(f) if should_curry(f) else f for name, f in ns.items() if '__' not in name}