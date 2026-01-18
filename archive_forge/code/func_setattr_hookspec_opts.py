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
def setattr_hookspec_opts(func: _F) -> _F:
    if historic and firstresult:
        raise ValueError('cannot have a historic firstresult hook')
    opts: HookspecOpts = {'firstresult': firstresult, 'historic': historic, 'warn_on_impl': warn_on_impl}
    setattr(func, self.project_name + '_spec', opts)
    return func