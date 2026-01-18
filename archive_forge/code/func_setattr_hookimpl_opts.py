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
def setattr_hookimpl_opts(func: _F) -> _F:
    opts: HookimplOpts = {'wrapper': wrapper, 'hookwrapper': hookwrapper, 'optionalhook': optionalhook, 'tryfirst': tryfirst, 'trylast': trylast, 'specname': specname}
    setattr(func, self.project_name + '_impl', opts)
    return func