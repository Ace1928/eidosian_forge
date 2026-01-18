import dill
from functools import partial
import warnings
def collection_function_recursion():
    d = {}

    def g():
        return d
    d['g'] = g
    return g