from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
def identity_key_from_row(self, row: Optional[Union[Row[Any], RowMapping]], identity_token: Optional[Any]=None, adapter: Optional[ORMAdapter]=None) -> _IdentityKeyType[_O]:
    """Return an identity-map key for use in storing/retrieving an
        item from the identity map.

        :param row: A :class:`.Row` or :class:`.RowMapping` produced from a
         result set that selected from the ORM mapped primary key columns.

         .. versionchanged:: 2.0
            :class:`.Row` or :class:`.RowMapping` are accepted
            for the "row" argument

        """
    pk_cols: Sequence[ColumnClause[Any]] = self.primary_key
    if adapter:
        pk_cols = [adapter.columns[c] for c in pk_cols]
    if hasattr(row, '_mapping'):
        mapping = row._mapping
    else:
        mapping = cast('Mapping[Any, Any]', row)
    return (self._identity_class, tuple((mapping[column] for column in pk_cols)), identity_token)