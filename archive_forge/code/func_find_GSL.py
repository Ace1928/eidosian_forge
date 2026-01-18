import logging
import platform
from pyomo.common import Library
from pyomo.common.deprecation import deprecated
def find_GSL():
    if platform.python_implementation().lower().startswith('pypy'):
        return None
    return Library('amplgsl.dll').path()