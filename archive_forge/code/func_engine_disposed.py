from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from .base import Connection
from .base import Engine
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPIConnection
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .. import event
from .. import exc
from ..util.typing import Literal
def engine_disposed(self, engine: Engine) -> None:
    """Intercept when the :meth:`_engine.Engine.dispose` method is called.

        The :meth:`_engine.Engine.dispose` method instructs the engine to
        "dispose" of it's connection pool (e.g. :class:`_pool.Pool`), and
        replaces it with a new one.  Disposing of the old pool has the
        effect that existing checked-in connections are closed.  The new
        pool does not establish any new connections until it is first used.

        This event can be used to indicate that resources related to the
        :class:`_engine.Engine` should also be cleaned up,
        keeping in mind that the
        :class:`_engine.Engine`
        can still be used for new requests in which case
        it re-acquires connection resources.

        """