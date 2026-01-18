from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def set_use_numexpr(v: bool=True) -> None:
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v
    global _evaluate, _where
    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard