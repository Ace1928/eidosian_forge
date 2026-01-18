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
def from_select(self, names: Sequence[_DMLColumnArgument], select: Selectable, include_defaults: bool=True) -> Self:
    """Return a new :class:`_expression.Insert` construct which represents
        an ``INSERT...FROM SELECT`` statement.

        e.g.::

            sel = select(table1.c.a, table1.c.b).where(table1.c.c > 5)
            ins = table2.insert().from_select(['a', 'b'], sel)

        :param names: a sequence of string column names or
         :class:`_schema.Column`
         objects representing the target columns.
        :param select: a :func:`_expression.select` construct,
         :class:`_expression.FromClause`
         or other construct which resolves into a
         :class:`_expression.FromClause`,
         such as an ORM :class:`_query.Query` object, etc.  The order of
         columns returned from this FROM clause should correspond to the
         order of columns sent as the ``names`` parameter;  while this
         is not checked before passing along to the database, the database
         would normally raise an exception if these column lists don't
         correspond.
        :param include_defaults: if True, non-server default values and
         SQL expressions as specified on :class:`_schema.Column` objects
         (as documented in :ref:`metadata_defaults_toplevel`) not
         otherwise specified in the list of names will be rendered
         into the INSERT and SELECT statements, so that these values are also
         included in the data to be inserted.

         .. note:: A Python-side default that uses a Python callable function
            will only be invoked **once** for the whole statement, and **not
            per row**.

        """
    if self._values:
        raise exc.InvalidRequestError('This construct already inserts value expressions')
    self._select_names = [coercions.expect(roles.DMLColumnRole, name, as_key=True) for name in names]
    self._inline = True
    self.include_insert_from_select_defaults = include_defaults
    self.select = coercions.expect(roles.DMLSelectRole, select)
    return self