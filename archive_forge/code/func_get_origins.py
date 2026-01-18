import operator
import sys
from functools import reduce
from importlib import import_module
from types import ModuleType
def get_origins(defs):
    origins = {}
    for module, attrs in defs.items():
        origins.update({attr: module for attr in attrs})
    return origins