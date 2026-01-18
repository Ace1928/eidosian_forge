from __future__ import annotations
import typing as T
from ...interpreterbase import (
@noKwargs
@noPosargs
def keys_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> T.List[str]:
    return sorted(self.held_object)