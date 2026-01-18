from __future__ import annotations
import contextlib
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from .interfaces import BindTyping
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPICursor
from .interfaces import ExceptionContext
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .interfaces import IsolationLevel
from .util import _distill_params_20
from .util import _distill_raw_params
from .util import TransactionalContext
from .. import exc
from .. import inspection
from .. import log
from .. import util
from ..sql import compiler
from ..sql import util as sql_util
def exec_driver_sql(self, statement: str, parameters: Optional[_DBAPIAnyExecuteParams]=None, execution_options: Optional[CoreExecuteOptionsParameter]=None) -> CursorResult[Any]:
    """Executes a string SQL statement on the DBAPI cursor directly,
        without any SQL compilation steps.

        This can be used to pass any string directly to the
        ``cursor.execute()`` method of the DBAPI in use.

        :param statement: The statement str to be executed.   Bound parameters
         must use the underlying DBAPI's paramstyle, such as "qmark",
         "pyformat", "format", etc.

        :param parameters: represent bound parameter values to be used in the
         execution.  The format is one of:   a dictionary of named parameters,
         a tuple of positional parameters, or a list containing either
         dictionaries or tuples for multiple-execute support.

        :return: a :class:`_engine.CursorResult`.

         E.g. multiple dictionaries::


             conn.exec_driver_sql(
                 "INSERT INTO table (id, value) VALUES (%(id)s, %(value)s)",
                 [{"id":1, "value":"v1"}, {"id":2, "value":"v2"}]
             )

         Single dictionary::

             conn.exec_driver_sql(
                 "INSERT INTO table (id, value) VALUES (%(id)s, %(value)s)",
                 dict(id=1, value="v1")
             )

         Single tuple::

             conn.exec_driver_sql(
                 "INSERT INTO table (id, value) VALUES (?, ?)",
                 (1, 'v1')
             )

         .. note:: The :meth:`_engine.Connection.exec_driver_sql` method does
             not participate in the
             :meth:`_events.ConnectionEvents.before_execute` and
             :meth:`_events.ConnectionEvents.after_execute` events.   To
             intercept calls to :meth:`_engine.Connection.exec_driver_sql`, use
             :meth:`_events.ConnectionEvents.before_cursor_execute` and
             :meth:`_events.ConnectionEvents.after_cursor_execute`.

         .. seealso::

            :pep:`249`

        """
    distilled_parameters = _distill_raw_params(parameters)
    execution_options = self._execution_options.merge_with(execution_options)
    dialect = self.dialect
    ret = self._execute_context(dialect, dialect.execution_ctx_cls._init_statement, statement, None, execution_options, statement, distilled_parameters)
    return ret