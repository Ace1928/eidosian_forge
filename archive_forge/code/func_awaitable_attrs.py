from __future__ import annotations
import asyncio
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import engine
from .base import ReversibleProxy
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import util
from ...orm import close_all_sessions as _sync_close_all_sessions
from ...orm import object_session
from ...orm import Session
from ...orm import SessionTransaction
from ...orm import state as _instance_state
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
@property
def awaitable_attrs(self) -> AsyncAttrs._AsyncAttrGetitem:
    """provide a namespace of all attributes on this object wrapped
        as awaitables.

        e.g.::


            a1 = (await async_session.scalars(select(A).where(A.id == 5))).one()

            some_attribute = await a1.awaitable_attrs.some_deferred_attribute
            some_collection = await a1.awaitable_attrs.some_collection

        """
    return AsyncAttrs._AsyncAttrGetitem(self)