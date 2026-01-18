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
@classmethod
@util.deprecated('1.4', 'The from_engine() method on :class:`_reflection.Inspector` is deprecated and will be removed in a future release.  Please use the :func:`.sqlalchemy.inspect` function on an :class:`_engine.Engine` or :class:`_engine.Connection` in order to acquire an :class:`_reflection.Inspector`.')
def from_engine(cls, bind: Engine) -> Inspector:
    """Construct a new dialect-specific Inspector object from the given
        engine or connection.

        :param bind: a :class:`~sqlalchemy.engine.Connection`
         or :class:`~sqlalchemy.engine.Engine`.

        This method differs from direct a direct constructor call of
        :class:`_reflection.Inspector` in that the
        :class:`~sqlalchemy.engine.interfaces.Dialect` is given a chance to
        provide a dialect-specific :class:`_reflection.Inspector` instance,
        which may
        provide additional methods.

        See the example at :class:`_reflection.Inspector`.

        """
    return cls._construct(cls._init_legacy, bind)