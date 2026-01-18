import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def dynamic_redim(obj, **dynkwargs):
    return obj.redim(specs, **dimensions)