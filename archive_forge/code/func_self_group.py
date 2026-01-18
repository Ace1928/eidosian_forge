from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TypeVar
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql._typing import _TypeEngineArgument
def self_group(self, against=None):
    if against in (operators.any_op, operators.all_op, operators.getitem):
        return expression.Grouping(self)
    else:
        return self