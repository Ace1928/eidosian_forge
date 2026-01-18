from __future__ import annotations
import inspect
import sys
import warnings
from types import ModuleType
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Final
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar
from typing import Union
from ._result import Result
def call_historic(self, result_callback: Callable[[Any], None] | None=None, kwargs: Mapping[str, object] | None=None) -> None:
    """Call the hook with given ``kwargs`` for all registered plugins and
        for all plugins which will be registered afterwards, see
        :ref:`historic`.

        :param result_callback:
            If provided, will be called for each non-``None`` result obtained
            from a hook implementation.
        """
    assert self._call_history is not None
    kwargs = kwargs or {}
    self._verify_all_args_are_provided(kwargs)
    self._call_history.append((kwargs, result_callback))
    res = self._hookexec(self.name, self._hookimpls.copy(), kwargs, False)
    if result_callback is None:
        return
    if isinstance(res, list):
        for x in res:
            result_callback(x)