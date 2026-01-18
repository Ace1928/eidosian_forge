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
def compare_values(self, x, y):
    return x == y