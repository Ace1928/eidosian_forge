import pyomo.common.unittest as unittest
import sys
from importlib import import_module
from io import StringIO
from pyomo.common.log import LoggingIntercept
def force_load(module):
    if module in sys.modules:
        del sys.modules[module]
    return import_module(module)