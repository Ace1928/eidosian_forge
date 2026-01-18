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
@event._legacy_signature('1.4', ['conn', 'clauseelement', 'multiparams', 'params'], lambda conn, clauseelement, multiparams, params, execution_options: (conn, clauseelement, multiparams, params))
def before_execute(self, conn: Connection, clauseelement: Executable, multiparams: _CoreMultiExecuteParams, params: _CoreSingleExecuteParams, execution_options: _ExecuteOptions) -> Optional[Tuple[Executable, _CoreMultiExecuteParams, _CoreSingleExecuteParams]]:
    """Intercept high level execute() events, receiving uncompiled
        SQL constructs and other objects prior to rendering into SQL.

        This event is good for debugging SQL compilation issues as well
        as early manipulation of the parameters being sent to the database,
        as the parameter lists will be in a consistent format here.

        This event can be optionally established with the ``retval=True``
        flag.  The ``clauseelement``, ``multiparams``, and ``params``
        arguments should be returned as a three-tuple in this case::

            @event.listens_for(Engine, "before_execute", retval=True)
            def before_execute(conn, clauseelement, multiparams, params):
                # do something with clauseelement, multiparams, params
                return clauseelement, multiparams, params

        :param conn: :class:`_engine.Connection` object
        :param clauseelement: SQL expression construct, :class:`.Compiled`
         instance, or string statement passed to
         :meth:`_engine.Connection.execute`.
        :param multiparams: Multiple parameter sets, a list of dictionaries.
        :param params: Single parameter set, a single dictionary.
        :param execution_options: dictionary of execution
         options passed along with the statement, if any.  This is a merge
         of all options that will be used, including those of the statement,
         the connection, and those passed in to the method itself for
         the 2.0 style of execution.

         .. versionadded: 1.4

        .. seealso::

            :meth:`.before_cursor_execute`

        """