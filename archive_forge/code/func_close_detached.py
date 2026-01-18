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
def close_detached(self, dbapi_connection: DBAPIConnection) -> None:
    """Called when a detached DBAPI connection is closed.

        The event is emitted before the close occurs.

        The close of a connection can fail; typically this is because
        the connection is already closed.  If the close operation fails,
        the connection is discarded.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        """