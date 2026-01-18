from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
def flexi_cache(*traverse_args: Tuple[str, InternalTraversal]) -> Callable[[Callable[..., _R]], Callable[..., _R]]:

    @util.decorator
    def go(fn: Callable[..., _R], self: Dialect, con: Connection, *args: Any, **kw: Any) -> _R:
        info_cache = kw.get('info_cache', None)
        if info_cache is None:
            return fn(self, con, *args, **kw)
        key = _ad_hoc_cache_key_from_args((fn.__name__,), traverse_args, args)
        ret: _R = info_cache.get(key)
        if ret is None:
            ret = fn(self, con, *args, **kw)
            info_cache[key] = ret
        return ret
    return go