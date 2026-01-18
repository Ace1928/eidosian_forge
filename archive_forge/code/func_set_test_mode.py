from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def set_test_mode(v: bool=True) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
    global _TEST_MODE, _TEST_RESULT
    _TEST_MODE = v
    _TEST_RESULT = []