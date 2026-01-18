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
def do_connect(self, dialect: Dialect, conn_rec: ConnectionPoolEntry, cargs: Tuple[Any, ...], cparams: Dict[str, Any]) -> Optional[DBAPIConnection]:
    """Receive connection arguments before a connection is made.

        This event is useful in that it allows the handler to manipulate the
        cargs and/or cparams collections that control how the DBAPI
        ``connect()`` function will be called. ``cargs`` will always be a
        Python list that can be mutated in-place, and ``cparams`` a Python
        dictionary that may also be mutated::

            e = create_engine("postgresql+psycopg2://user@host/dbname")

            @event.listens_for(e, 'do_connect')
            def receive_do_connect(dialect, conn_rec, cargs, cparams):
                cparams["password"] = "some_password"

        The event hook may also be used to override the call to ``connect()``
        entirely, by returning a non-``None`` DBAPI connection object::

            e = create_engine("postgresql+psycopg2://user@host/dbname")

            @event.listens_for(e, 'do_connect')
            def receive_do_connect(dialect, conn_rec, cargs, cparams):
                return psycopg2.connect(*cargs, **cparams)

        .. seealso::

            :ref:`custom_dbapi_args`

        """