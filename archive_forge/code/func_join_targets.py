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
def join_targets(self, source_selectable: Optional[FromClause], dest_selectable: FromClause, aliased: bool, single_crit: Optional[ColumnElement[bool]]=None, extra_criteria: Tuple[ColumnElement[bool], ...]=()) -> Tuple[ColumnElement[bool], Optional[ColumnElement[bool]], Optional[FromClause], Optional[ClauseAdapter], FromClause]:
    """Given a source and destination selectable, create a
        join between them.

        This takes into account aliasing the join clause
        to reference the appropriate corresponding columns
        in the target objects, as well as the extra child
        criterion, equivalent column sets, etc.

        """
    dest_selectable = _shallow_annotate(dest_selectable, {'no_replacement_traverse': True})
    primaryjoin, secondaryjoin, secondary = (self.primaryjoin, self.secondaryjoin, self.secondary)
    if single_crit is not None:
        if secondaryjoin is not None:
            secondaryjoin = secondaryjoin & single_crit
        else:
            primaryjoin = primaryjoin & single_crit
    if extra_criteria:

        def mark_exclude_cols(elem: SupportsAnnotations, annotations: _AnnotationDict) -> SupportsAnnotations:
            """note unrelated columns in the "extra criteria" as either
                should be adapted or not adapted, even though they are not
                part of our "local" or "remote" side.

                see #9779 for this case, as well as #11010 for a follow up

                """
            parentmapper_for_element = elem._annotations.get('parentmapper', None)
            if parentmapper_for_element is not self.prop.parent and parentmapper_for_element is not self.prop.mapper and (elem not in self._secondary_lineage_set):
                return _safe_annotate(elem, annotations)
            else:
                return elem
        extra_criteria = tuple((_deep_annotate(elem, {'should_not_adapt': True}, annotate_callable=mark_exclude_cols) for elem in extra_criteria))
        if secondaryjoin is not None:
            secondaryjoin = secondaryjoin & sql.and_(*extra_criteria)
        else:
            primaryjoin = primaryjoin & sql.and_(*extra_criteria)
    if aliased:
        if secondary is not None:
            secondary = secondary._anonymous_fromclause(flat=True)
            primary_aliasizer = ClauseAdapter(secondary, exclude_fn=_local_col_exclude)
            secondary_aliasizer = ClauseAdapter(dest_selectable, equivalents=self.child_equivalents).chain(primary_aliasizer)
            if source_selectable is not None:
                primary_aliasizer = ClauseAdapter(secondary, exclude_fn=_local_col_exclude).chain(ClauseAdapter(source_selectable, equivalents=self.parent_equivalents))
            secondaryjoin = secondary_aliasizer.traverse(secondaryjoin)
        else:
            primary_aliasizer = ClauseAdapter(dest_selectable, exclude_fn=_local_col_exclude, equivalents=self.child_equivalents)
            if source_selectable is not None:
                primary_aliasizer.chain(ClauseAdapter(source_selectable, exclude_fn=_remote_col_exclude, equivalents=self.parent_equivalents))
            secondary_aliasizer = None
        primaryjoin = primary_aliasizer.traverse(primaryjoin)
        target_adapter = secondary_aliasizer or primary_aliasizer
        target_adapter.exclude_fn = None
    else:
        target_adapter = None
    return (primaryjoin, secondaryjoin, secondary, target_adapter, dest_selectable)