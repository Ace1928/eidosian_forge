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
def do_setinputsizes(self, inputsizes: Dict[BindParameter[Any], Any], cursor: DBAPICursor, statement: str, parameters: _DBAPIAnyExecuteParams, context: ExecutionContext) -> None:
    """Receive the setinputsizes dictionary for possible modification.

        This event is emitted in the case where the dialect makes use of the
        DBAPI ``cursor.setinputsizes()`` method which passes information about
        parameter binding for a particular statement.   The given
        ``inputsizes`` dictionary will contain :class:`.BindParameter` objects
        as keys, linked to DBAPI-specific type objects as values; for
        parameters that are not bound, they are added to the dictionary with
        ``None`` as the value, which means the parameter will not be included
        in the ultimate setinputsizes call.   The event may be used to inspect
        and/or log the datatypes that are being bound, as well as to modify the
        dictionary in place.  Parameters can be added, modified, or removed
        from this dictionary.   Callers will typically want to inspect the
        :attr:`.BindParameter.type` attribute of the given bind objects in
        order to make decisions about the DBAPI object.

        After the event, the ``inputsizes`` dictionary is converted into
        an appropriate datastructure to be passed to ``cursor.setinputsizes``;
        either a list for a positional bound parameter execution style,
        or a dictionary of string parameter keys to DBAPI type objects for
        a named bound parameter execution style.

        The setinputsizes hook overall is only used for dialects which include
        the flag ``use_setinputsizes=True``.  Dialects which use this
        include cx_Oracle, pg8000, asyncpg, and pyodbc dialects.

        .. note::

            For use with pyodbc, the ``use_setinputsizes`` flag
            must be passed to the dialect, e.g.::

                create_engine("mssql+pyodbc://...", use_setinputsizes=True)

            .. seealso::

                  :ref:`mssql_pyodbc_setinputsizes`

        .. versionadded:: 1.2.9

        .. seealso::

            :ref:`cx_oracle_setinputsizes`

        """
    pass