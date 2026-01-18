from __future__ import annotations
import typing
from typing import Any
from typing import Optional
from typing import Type
from typing import Union
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .base import PoolResetState
from .. import event
from .. import util
def soft_invalidate(self, dbapi_connection: DBAPIConnection, connection_record: ConnectionPoolEntry, exception: Optional[BaseException]) -> None:
    """Called when a DBAPI connection is to be "soft invalidated".

        This event is called any time the
        :meth:`.ConnectionPoolEntry.invalidate`
        method is invoked with the ``soft`` flag.

        Soft invalidation refers to when the connection record that tracks
        this connection will force a reconnect after the current connection
        is checked in.   It does not actively close the dbapi_connection
        at the point at which it is called.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        :param exception: the exception object corresponding to the reason
         for this invalidation, if any.  May be ``None``.

        """