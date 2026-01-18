from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import mapperlib as mapperlib
from ._typing import _O
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .interfaces import _AttributeOptions
from .properties import MappedColumn
from .properties import MappedSQLExpression
from .query import AliasOption
from .relationships import _RelationshipArgumentType
from .relationships import _RelationshipDeclared
from .relationships import _RelationshipSecondaryArgument
from .relationships import RelationshipProperty
from .session import Session
from .util import _ORMJoin
from .util import AliasedClass
from .util import AliasedInsp
from .util import LoaderCriteriaOption
from .. import sql
from .. import util
from ..exc import InvalidRequestError
from ..sql._typing import _no_kw
from ..sql.base import _NoArg
from ..sql.base import SchemaEventTarget
from ..sql.schema import _InsertSentinelColumnDefault
from ..sql.schema import SchemaConst
from ..sql.selectable import FromClause
from ..util.typing import Annotated
from ..util.typing import Literal
def query_expression(default_expr: _ORMColumnExprArgument[_T]=sql.null(), *, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, compare: Union[_NoArg, bool]=_NoArg.NO_ARG, expire_on_flush: bool=True, info: Optional[_InfoType]=None, doc: Optional[str]=None) -> MappedSQLExpression[_T]:
    """Indicate an attribute that populates from a query-time SQL expression.

    :param default_expr: Optional SQL expression object that will be used in
        all cases if not assigned later with :func:`_orm.with_expression`.

    .. versionadded:: 1.2

    .. seealso::

        :ref:`orm_queryguide_with_expression` - background and usage examples

    """
    prop = MappedSQLExpression(default_expr, attribute_options=_AttributeOptions(False, repr, _NoArg.NO_ARG, _NoArg.NO_ARG, compare, _NoArg.NO_ARG), expire_on_flush=expire_on_flush, info=info, doc=doc, _assume_readonly_dc_attributes=True)
    prop.strategy_key = (('query_expression', True),)
    return prop