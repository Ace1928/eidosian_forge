from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from . import util as orm_util
from ._typing import _O
from .base import _assertions
from .context import _column_descriptions
from .context import _determine_last_joined_entity
from .context import _legacy_filter_by_entity_zero
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .util import AliasedClass
from .util import object_mapper
from .util import with_parent
from .. import exc as sa_exc
from .. import inspect
from .. import inspection
from .. import log
from .. import sql
from .. import util
from ..engine import Result
from ..engine import Row
from ..event import dispatcher
from ..event import EventTarget
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import Select
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _FromClauseArgument
from ..sql._typing import _TP
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import _entity_namespace_key
from ..sql.base import _generative
from ..sql.base import _NoArg
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.elements import BooleanClauseList
from ..sql.expression import Exists
from ..sql.selectable import _MemoizedSelectEntities
from ..sql.selectable import _SelectFromElements
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import HasHints
from ..sql.selectable import HasPrefixes
from ..sql.selectable import HasSuffixes
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectLabelStyle
from ..util.typing import Literal
from ..util.typing import Self
def set_label_style(self, style: SelectLabelStyle) -> Self:
    """Apply column labels to the return value of Query.statement.

        Indicates that this Query's `statement` accessor should return
        a SELECT statement that applies labels to all columns in the
        form <tablename>_<columnname>; this is commonly used to
        disambiguate columns from multiple tables which have the same
        name.

        When the `Query` actually issues SQL to load rows, it always
        uses column labeling.

        .. note:: The :meth:`_query.Query.set_label_style` method *only* applies
           the output of :attr:`_query.Query.statement`, and *not* to any of
           the result-row invoking systems of :class:`_query.Query` itself,
           e.g.
           :meth:`_query.Query.first`, :meth:`_query.Query.all`, etc.
           To execute
           a query using :meth:`_query.Query.set_label_style`, invoke the
           :attr:`_query.Query.statement` using :meth:`.Session.execute`::

                result = session.execute(
                    query
                    .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
                    .statement
                )

        .. versionadded:: 1.4


        .. seealso::

            :meth:`_sql.Select.set_label_style` - v2 equivalent method.

        """
    if self._label_style is not style:
        self = self._generate()
        self._label_style = style
    return self