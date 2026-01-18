from abc import ABC, abstractmethod
from builtins import bool
from typing import Any, Dict, List, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
def basic_unary_arithmetic_op(self, col: Column, op: str) -> Column:
    if op == '+':
        return col
    if op == '-':
        return Column(0 - col.native)
    raise NotImplementedError(f'{op} is not supported')