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
def is_valid_private_name(name: str) -> bool:
    return not is_valid_field(name) and name not in DUNDER_ATTRIBUTES