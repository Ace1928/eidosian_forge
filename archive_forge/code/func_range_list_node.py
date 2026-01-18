from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def range_list_node(range_list: Iterable[Iterable[float | decimal.Decimal]]) -> tuple[Literal['range_list'], Iterable[Iterable[float | decimal.Decimal]]]:
    return ('range_list', range_list)