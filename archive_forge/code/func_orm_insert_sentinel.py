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
def orm_insert_sentinel(name: Optional[str]=None, type_: Optional[_TypeEngineArgument[Any]]=None, *, default: Optional[Any]=None, omit_from_statements: bool=True) -> MappedColumn[Any]:
    """Provides a surrogate :func:`_orm.mapped_column` that generates
    a so-called :term:`sentinel` column, allowing efficient bulk
    inserts with deterministic RETURNING sorting for tables that don't
    otherwise have qualifying primary key configurations.

    Use of :func:`_orm.orm_insert_sentinel` is analogous to the use of the
    :func:`_schema.insert_sentinel` construct within a Core
    :class:`_schema.Table` construct.

    Guidelines for adding this construct to a Declarative mapped class
    are the same as that of the :func:`_schema.insert_sentinel` construct;
    the database table itself also needs to have a column with this name
    present.

    For background on how this object is used, see the section
    :ref:`engine_insertmanyvalues_sentinel_columns` as part of the
    section :ref:`engine_insertmanyvalues`.

    .. seealso::

        :func:`_schema.insert_sentinel`

        :ref:`engine_insertmanyvalues`

        :ref:`engine_insertmanyvalues_sentinel_columns`


    .. versionadded:: 2.0.10

    """
    return mapped_column(name=name, default=default if default is not None else _InsertSentinelColumnDefault(), _omit_from_statements=omit_from_statements, insert_sentinel=True, use_existing_column=True, nullable=True)