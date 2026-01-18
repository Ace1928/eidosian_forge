from __future__ import annotations
from ...interpreterbase import (
import typing as T
@noKwargs
@noPosargs
def is_even_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> bool:
    return self.held_object % 2 == 0