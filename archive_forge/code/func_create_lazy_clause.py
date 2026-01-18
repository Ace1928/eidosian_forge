from __future__ import annotations
import collections
from collections import abc
import dataclasses
import inspect as _py_inspect
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import strategy_options
from ._typing import insp_is_aliased_class
from ._typing import is_has_collection_adapter
from .base import _DeclarativeMapped
from .base import _is_mapped_class
from .base import class_mapper
from .base import DynamicMapped
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .base import state_str
from .base import WriteOnlyMapped
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .interfaces import PropComparator
from .interfaces import RelationshipDirection
from .interfaces import StrategizedProperty
from .util import _orm_annotate
from .util import _orm_deannotate
from .util import CascadeOptions
from .. import exc as sa_exc
from .. import Exists
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..inspection import inspect
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql._typing import _ColumnExpressionArgument
from ..sql._typing import _HasClauseElement
from ..sql.annotation import _safe_annotate
from ..sql.elements import ColumnClause
from ..sql.elements import ColumnElement
from ..sql.util import _deep_annotate
from ..sql.util import _deep_deannotate
from ..sql.util import _shallow_annotate
from ..sql.util import adapt_criterion_to_null
from ..sql.util import ClauseAdapter
from ..sql.util import join_condition
from ..sql.util import selectables_overlap
from ..sql.util import visit_binary_product
from ..util.typing import de_optionalize_union_types
from ..util.typing import Literal
from ..util.typing import resolve_name_to_real_class_name
def create_lazy_clause(self, reverse_direction: bool=False) -> Tuple[ColumnElement[bool], Dict[str, ColumnElement[Any]], Dict[ColumnElement[Any], ColumnElement[Any]]]:
    binds: Dict[ColumnElement[Any], BindParameter[Any]] = {}
    equated_columns: Dict[ColumnElement[Any], ColumnElement[Any]] = {}
    has_secondary = self.secondaryjoin is not None
    if has_secondary:
        lookup = collections.defaultdict(list)
        for l, r in self.local_remote_pairs:
            lookup[l].append((l, r))
            equated_columns[r] = l
    elif not reverse_direction:
        for l, r in self.local_remote_pairs:
            equated_columns[r] = l
    else:
        for l, r in self.local_remote_pairs:
            equated_columns[l] = r

    def col_to_bind(element: ColumnElement[Any], **kw: Any) -> Optional[BindParameter[Any]]:
        if not reverse_direction and 'local' in element._annotations or (reverse_direction and (has_secondary and element in lookup or (not has_secondary and 'remote' in element._annotations))):
            if element not in binds:
                binds[element] = sql.bindparam(None, None, type_=element.type, unique=True)
            return binds[element]
        return None
    lazywhere = self.primaryjoin
    if self.secondaryjoin is None or not reverse_direction:
        lazywhere = visitors.replacement_traverse(lazywhere, {}, col_to_bind)
    if self.secondaryjoin is not None:
        secondaryjoin = self.secondaryjoin
        if reverse_direction:
            secondaryjoin = visitors.replacement_traverse(secondaryjoin, {}, col_to_bind)
        lazywhere = sql.and_(lazywhere, secondaryjoin)
    bind_to_col = {binds[col].key: col for col in binds}
    return (lazywhere, bind_to_col, equated_columns)