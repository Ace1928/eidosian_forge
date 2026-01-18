import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def nodata(self, specs=None, **values):
    return self._redim('nodata', specs, **values)