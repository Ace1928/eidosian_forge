from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def get_code_object(obj):
    """Shamelessly borrowed from llpython"""
    return getattr(obj, '__code__', getattr(obj, 'func_code', None))