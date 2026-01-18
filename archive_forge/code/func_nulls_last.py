from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple as typing_Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from .base import _NoArg
from .coercions import _document_text_coercion
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import Case
from .elements import Cast
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Extract
from .elements import False_
from .elements import FunctionFilter
from .elements import Label
from .elements import Null
from .elements import Over
from .elements import TextClause
from .elements import True_
from .elements import TryCast
from .elements import Tuple
from .elements import TypeCoerce
from .elements import UnaryExpression
from .elements import WithinGroup
from .functions import FunctionElement
from ..util.typing import Literal
def nulls_last(column: _ColumnExpressionArgument[_T]) -> UnaryExpression[_T]:
    """Produce the ``NULLS LAST`` modifier for an ``ORDER BY`` expression.

    :func:`.nulls_last` is intended to modify the expression produced
    by :func:`.asc` or :func:`.desc`, and indicates how NULL values
    should be handled when they are encountered during ordering::


        from sqlalchemy import desc, nulls_last

        stmt = select(users_table).order_by(
            nulls_last(desc(users_table.c.name)))

    The SQL expression from the above would resemble::

        SELECT id, name FROM user ORDER BY name DESC NULLS LAST

    Like :func:`.asc` and :func:`.desc`, :func:`.nulls_last` is typically
    invoked from the column expression itself using
    :meth:`_expression.ColumnElement.nulls_last`,
    rather than as its standalone
    function version, as in::

        stmt = select(users_table).order_by(
            users_table.c.name.desc().nulls_last())

    .. versionchanged:: 1.4 :func:`.nulls_last` is renamed from
        :func:`.nullslast` in previous releases.
        The previous name remains available for backwards compatibility.

    .. seealso::

        :func:`.asc`

        :func:`.desc`

        :func:`.nulls_first`

        :meth:`_expression.Select.order_by`

    """
    return UnaryExpression._create_nulls_last(column)