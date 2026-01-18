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
def backref(name: str, **kwargs: Any) -> ORMBackrefArgument:
    """When using the :paramref:`_orm.relationship.backref` parameter,
    provides specific parameters to be used when the new
    :func:`_orm.relationship` is generated.

    E.g.::

        'items':relationship(
            SomeItem, backref=backref('parent', lazy='subquery'))

    The :paramref:`_orm.relationship.backref` parameter is generally
    considered to be legacy; for modern applications, using
    explicit :func:`_orm.relationship` constructs linked together using
    the :paramref:`_orm.relationship.back_populates` parameter should be
    preferred.

    .. seealso::

        :ref:`relationships_backref` - background on backrefs

    """
    return (name, kwargs)