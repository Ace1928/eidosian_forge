from __future__ import annotations
import sys
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import TypeVar
import_prefix = _reg.import_prefix
def preload_module(self, *deps: str) -> Callable[[_FN], _FN]:
    """Adds the specified modules to the list to load.

        This method can be used both as a normal function and as a decorator.
        No change is performed to the decorated object.
        """
    self.module_registry.update(deps)
    return lambda fn: fn