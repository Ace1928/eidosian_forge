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
def subquery(self, name: Optional[str]=None, with_labels: bool=False, reduce_columns: bool=False) -> Subquery:
    """Return the full SELECT statement represented by
        this :class:`_query.Query`, embedded within an
        :class:`_expression.Alias`.

        Eager JOIN generation within the query is disabled.

        .. seealso::

            :meth:`_sql.Select.subquery` - v2 comparable method.

        :param name: string name to be assigned as the alias;
            this is passed through to :meth:`_expression.FromClause.alias`.
            If ``None``, a name will be deterministically generated
            at compile time.

        :param with_labels: if True, :meth:`.with_labels` will be called
         on the :class:`_query.Query` first to apply table-qualified labels
         to all columns.

        :param reduce_columns: if True,
         :meth:`_expression.Select.reduce_columns` will
         be called on the resulting :func:`_expression.select` construct,
         to remove same-named columns where one also refers to the other
         via foreign key or WHERE clause equivalence.

        """
    q = self.enable_eagerloads(False)
    if with_labels:
        q = q.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
    stmt = q._get_select_statement_only()
    if TYPE_CHECKING:
        assert isinstance(stmt, Select)
    if reduce_columns:
        stmt = stmt.reduce_columns()
    return stmt.subquery(name=name)