from __future__ import annotations
import asyncio
import contextlib
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Dict
from typing import Generator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as async_exc
from .base import asyncstartablecontext
from .base import GeneratorStartableContext
from .base import ProxyComparable
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import exc
from ... import inspection
from ... import util
from ...engine import Connection
from ...engine import create_engine as _create_engine
from ...engine import create_pool_from_url as _create_pool_from_url
from ...engine import Engine
from ...engine.base import NestedTransaction
from ...engine.base import Transaction
from ...exc import ArgumentError
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
@property
def default_isolation_level(self) -> Any:
    """The initial-connection time isolation level associated with the
        :class:`_engine.Dialect` in use.

        .. container:: class_bases

            Proxied for the :class:`_engine.Connection` class
            on behalf of the :class:`_asyncio.AsyncConnection` class.

        This value is independent of the
        :paramref:`.Connection.execution_options.isolation_level` and
        :paramref:`.Engine.execution_options.isolation_level` execution
        options, and is determined by the :class:`_engine.Dialect` when the
        first connection is created, by performing a SQL query against the
        database for the current isolation level before any additional commands
        have been emitted.

        Calling this accessor does not invoke any new SQL queries.

        .. seealso::

            :meth:`_engine.Connection.get_isolation_level`
            - view current actual isolation level

            :paramref:`_sa.create_engine.isolation_level`
            - set per :class:`_engine.Engine` isolation level

            :paramref:`.Connection.execution_options.isolation_level`
            - set per :class:`_engine.Connection` isolation level


        """
    return self._proxied.default_isolation_level