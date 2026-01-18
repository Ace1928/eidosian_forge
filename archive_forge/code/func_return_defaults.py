from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
@_generative
def return_defaults(self, *cols: _DMLColumnArgument, supplemental_cols: Optional[Iterable[_DMLColumnArgument]]=None, sort_by_parameter_order: bool=False) -> Self:
    """Make use of a :term:`RETURNING` clause for the purpose
        of fetching server-side expressions and defaults, for supporting
        backends only.

        .. deepalchemy::

            The :meth:`.UpdateBase.return_defaults` method is used by the ORM
            for its internal work in fetching newly generated primary key
            and server default values, in particular to provide the underyling
            implementation of the :paramref:`_orm.Mapper.eager_defaults`
            ORM feature as well as to allow RETURNING support with bulk
            ORM inserts.  Its behavior is fairly idiosyncratic
            and is not really intended for general use.  End users should
            stick with using :meth:`.UpdateBase.returning` in order to
            add RETURNING clauses to their INSERT, UPDATE and DELETE
            statements.

        Normally, a single row INSERT statement will automatically populate the
        :attr:`.CursorResult.inserted_primary_key` attribute when executed,
        which stores the primary key of the row that was just inserted in the
        form of a :class:`.Row` object with column names as named tuple keys
        (and the :attr:`.Row._mapping` view fully populated as well). The
        dialect in use chooses the strategy to use in order to populate this
        data; if it was generated using server-side defaults and / or SQL
        expressions, dialect-specific approaches such as ``cursor.lastrowid``
        or ``RETURNING`` are typically used to acquire the new primary key
        value.

        However, when the statement is modified by calling
        :meth:`.UpdateBase.return_defaults` before executing the statement,
        additional behaviors take place **only** for backends that support
        RETURNING and for :class:`.Table` objects that maintain the
        :paramref:`.Table.implicit_returning` parameter at its default value of
        ``True``. In these cases, when the :class:`.CursorResult` is returned
        from the statement's execution, not only will
        :attr:`.CursorResult.inserted_primary_key` be populated as always, the
        :attr:`.CursorResult.returned_defaults` attribute will also be
        populated with a :class:`.Row` named-tuple representing the full range
        of server generated
        values from that single row, including values for any columns that
        specify :paramref:`_schema.Column.server_default` or which make use of
        :paramref:`_schema.Column.default` using a SQL expression.

        When invoking INSERT statements with multiple rows using
        :ref:`insertmanyvalues <engine_insertmanyvalues>`, the
        :meth:`.UpdateBase.return_defaults` modifier will have the effect of
        the :attr:`_engine.CursorResult.inserted_primary_key_rows` and
        :attr:`_engine.CursorResult.returned_defaults_rows` attributes being
        fully populated with lists of :class:`.Row` objects representing newly
        inserted primary key values as well as newly inserted server generated
        values for each row inserted. The
        :attr:`.CursorResult.inserted_primary_key` and
        :attr:`.CursorResult.returned_defaults` attributes will also continue
        to be populated with the first row of these two collections.

        If the backend does not support RETURNING or the :class:`.Table` in use
        has disabled :paramref:`.Table.implicit_returning`, then no RETURNING
        clause is added and no additional data is fetched, however the
        INSERT, UPDATE or DELETE statement proceeds normally.

        E.g.::

            stmt = table.insert().values(data='newdata').return_defaults()

            result = connection.execute(stmt)

            server_created_at = result.returned_defaults['created_at']

        When used against an UPDATE statement
        :meth:`.UpdateBase.return_defaults` instead looks for columns that
        include :paramref:`_schema.Column.onupdate` or
        :paramref:`_schema.Column.server_onupdate` parameters assigned, when
        constructing the columns that will be included in the RETURNING clause
        by default if explicit columns were not specified. When used against a
        DELETE statement, no columns are included in RETURNING by default, they
        instead must be specified explicitly as there are no columns that
        normally change values when a DELETE statement proceeds.

        .. versionadded:: 2.0  :meth:`.UpdateBase.return_defaults` is supported
           for DELETE statements also and has been moved from
           :class:`.ValuesBase` to :class:`.UpdateBase`.

        The :meth:`.UpdateBase.return_defaults` method is mutually exclusive
        against the :meth:`.UpdateBase.returning` method and errors will be
        raised during the SQL compilation process if both are used at the same
        time on one statement. The RETURNING clause of the INSERT, UPDATE or
        DELETE statement is therefore controlled by only one of these methods
        at a time.

        The :meth:`.UpdateBase.return_defaults` method differs from
        :meth:`.UpdateBase.returning` in these ways:

        1. :meth:`.UpdateBase.return_defaults` method causes the
           :attr:`.CursorResult.returned_defaults` collection to be populated
           with the first row from the RETURNING result. This attribute is not
           populated when using :meth:`.UpdateBase.returning`.

        2. :meth:`.UpdateBase.return_defaults` is compatible with existing
           logic used to fetch auto-generated primary key values that are then
           populated into the :attr:`.CursorResult.inserted_primary_key`
           attribute. By contrast, using :meth:`.UpdateBase.returning` will
           have the effect of the :attr:`.CursorResult.inserted_primary_key`
           attribute being left unpopulated.

        3. :meth:`.UpdateBase.return_defaults` can be called against any
           backend. Backends that don't support RETURNING will skip the usage
           of the feature, rather than raising an exception, *unless*
           ``supplemental_cols`` is passed. The return value
           of :attr:`_engine.CursorResult.returned_defaults` will be ``None``
           for backends that don't support RETURNING or for which the target
           :class:`.Table` sets :paramref:`.Table.implicit_returning` to
           ``False``.

        4. An INSERT statement invoked with executemany() is supported if the
           backend database driver supports the
           :ref:`insertmanyvalues <engine_insertmanyvalues>`
           feature which is now supported by most SQLAlchemy-included backends.
           When executemany is used, the
           :attr:`_engine.CursorResult.returned_defaults_rows` and
           :attr:`_engine.CursorResult.inserted_primary_key_rows` accessors
           will return the inserted defaults and primary keys.

           .. versionadded:: 1.4 Added
              :attr:`_engine.CursorResult.returned_defaults_rows` and
              :attr:`_engine.CursorResult.inserted_primary_key_rows` accessors.
              In version 2.0, the underlying implementation which fetches and
              populates the data for these attributes was generalized to be
              supported by most backends, whereas in 1.4 they were only
              supported by the ``psycopg2`` driver.


        :param cols: optional list of column key names or
         :class:`_schema.Column` that acts as a filter for those columns that
         will be fetched.
        :param supplemental_cols: optional list of RETURNING expressions,
          in the same form as one would pass to the
          :meth:`.UpdateBase.returning` method. When present, the additional
          columns will be included in the RETURNING clause, and the
          :class:`.CursorResult` object will be "rewound" when returned, so
          that methods like :meth:`.CursorResult.all` will return new rows
          mostly as though the statement used :meth:`.UpdateBase.returning`
          directly. However, unlike when using :meth:`.UpdateBase.returning`
          directly, the **order of the columns is undefined**, so can only be
          targeted using names or :attr:`.Row._mapping` keys; they cannot
          reliably be targeted positionally.

          .. versionadded:: 2.0

        :param sort_by_parameter_order: for a batch INSERT that is being
         executed against multiple parameter sets, organize the results of
         RETURNING so that the returned rows correspond to the order of
         parameter sets passed in.  This applies only to an :term:`executemany`
         execution for supporting dialects and typically makes use of the
         :term:`insertmanyvalues` feature.

         .. versionadded:: 2.0.10

         .. seealso::

            :ref:`engine_insertmanyvalues_returning_order` - background on
            sorting of RETURNING rows for bulk INSERT

        .. seealso::

            :meth:`.UpdateBase.returning`

            :attr:`_engine.CursorResult.returned_defaults`

            :attr:`_engine.CursorResult.returned_defaults_rows`

            :attr:`_engine.CursorResult.inserted_primary_key`

            :attr:`_engine.CursorResult.inserted_primary_key_rows`

        """
    if self._return_defaults:
        if self._return_defaults_columns and cols:
            self._return_defaults_columns = tuple(util.OrderedSet(self._return_defaults_columns).union((coercions.expect(roles.ColumnsClauseRole, c) for c in cols)))
        else:
            self._return_defaults_columns = ()
    else:
        self._return_defaults_columns = tuple((coercions.expect(roles.ColumnsClauseRole, c) for c in cols))
    self._return_defaults = True
    if sort_by_parameter_order:
        if not self.is_insert:
            raise exc.ArgumentError("The 'sort_by_parameter_order' argument to return_defaults() only applies to INSERT statements")
        self._sort_by_parameter_order = True
    if supplemental_cols:
        supplemental_col_tup = (coercions.expect(roles.ColumnsClauseRole, c) for c in supplemental_cols)
        if self._supplemental_returning is None:
            self._supplemental_returning = tuple(util.unique_list(supplemental_col_tup))
        else:
            self._supplemental_returning = tuple(util.unique_list(self._supplemental_returning + tuple(supplemental_col_tup)))
    return self