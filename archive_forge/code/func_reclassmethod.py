import operator
import sys
from functools import reduce
from importlib import import_module
from types import ModuleType
def reclassmethod(method):
    return classmethod(fun_of_method(method))