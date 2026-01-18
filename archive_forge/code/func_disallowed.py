from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
def disallowed(cls: type[_T]) -> type[_T]:
    cls.unsupported_nodes = ()
    for node in nodes:
        new_method = _node_not_implemented(node)
        name = f'visit_{node}'
        cls.unsupported_nodes += (name,)
        setattr(cls, name, new_method)
    return cls