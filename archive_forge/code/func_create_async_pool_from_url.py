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
def create_async_pool_from_url(url: Union[str, URL], **kwargs: Any) -> Pool:
    """Create a new async engine instance.

    Arguments passed to :func:`_asyncio.create_async_pool_from_url` are mostly
    identical to those passed to the :func:`_sa.create_pool_from_url` function.
    The specified dialect must be an asyncio-compatible dialect
    such as :ref:`dialect-postgresql-asyncpg`.

    .. versionadded:: 2.0.10

    """
    kwargs['_is_async'] = True
    return _create_pool_from_url(url, **kwargs)