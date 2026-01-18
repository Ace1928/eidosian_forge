import keyword
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import islice, zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import (
from typing_extensions import Annotated
from .errors import ConfigError
from .typing import (
from .version import version_info
def almost_equal_floats(value_1: float, value_2: float, *, delta: float=1e-08) -> bool:
    """
    Return True if two floats are almost equal
    """
    return abs(value_1 - value_2) <= delta