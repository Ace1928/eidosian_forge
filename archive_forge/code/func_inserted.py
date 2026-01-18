from __future__ import annotations
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from ... import exc
from ... import util
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import KeyedColumnElement
from ...sql.expression import alias
from ...sql.selectable import NamedFromClause
from ...util.typing import Self
@property
def inserted(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
    """Provide the "inserted" namespace for an ON DUPLICATE KEY UPDATE
        statement

        MySQL's ON DUPLICATE KEY UPDATE clause allows reference to the row
        that would be inserted, via a special function called ``VALUES()``.
        This attribute provides all columns in this row to be referenceable
        such that they will render within a ``VALUES()`` function inside the
        ON DUPLICATE KEY UPDATE clause.    The attribute is named ``.inserted``
        so as not to conflict with the existing
        :meth:`_expression.Insert.values` method.

        .. tip::  The :attr:`_mysql.Insert.inserted` attribute is an instance
            of :class:`_expression.ColumnCollection`, which provides an
            interface the same as that of the :attr:`_schema.Table.c`
            collection described at :ref:`metadata_tables_and_columns`.
            With this collection, ordinary names are accessible like attributes
            (e.g. ``stmt.inserted.some_column``), but special names and
            dictionary method names should be accessed using indexed access,
            such as ``stmt.inserted["column name"]`` or
            ``stmt.inserted["values"]``.  See the docstring for
            :class:`_expression.ColumnCollection` for further examples.

        .. seealso::

            :ref:`mysql_insert_on_duplicate_key_update` - example of how
            to use :attr:`_expression.Insert.inserted`

        """
    return self.inserted_alias.columns