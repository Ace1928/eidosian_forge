from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def set_use_bottleneck(v: bool=True) -> None:
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v