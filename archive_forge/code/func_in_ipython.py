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
def in_ipython() -> bool:
    """
    Check whether we're in an ipython environment, including jupyter notebooks.
    """
    try:
        eval('__IPYTHON__')
    except NameError:
        return False
    else:
        return True