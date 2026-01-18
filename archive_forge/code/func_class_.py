from __future__ import annotations
import enum
import functools
import re
import types
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes  # noqa
from . import exc
from ._typing import _O
from ._typing import insp_is_aliased_class
from ._typing import insp_is_mapper
from ._typing import prop_is_relationship
from .base import _class_to_mapper as _class_to_mapper
from .base import _MappedAnnotationBase
from .base import _never_set as _never_set  # noqa: F401
from .base import _none_set as _none_set  # noqa: F401
from .base import attribute_str as attribute_str  # noqa: F401
from .base import class_mapper as class_mapper
from .base import DynamicMapped
from .base import InspectionAttr as InspectionAttr
from .base import instance_str as instance_str  # noqa: F401
from .base import Mapped
from .base import object_mapper as object_mapper
from .base import object_state as object_state  # noqa: F401
from .base import opt_manager_of_class
from .base import ORMDescriptor
from .base import state_attribute_str as state_attribute_str  # noqa: F401
from .base import state_class_str as state_class_str  # noqa: F401
from .base import state_str as state_str  # noqa: F401
from .base import WriteOnlyMapped
from .interfaces import CriteriaOption
from .interfaces import MapperProperty as MapperProperty
from .interfaces import ORMColumnsClauseRole
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .path_registry import PathRegistry as PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import sql
from .. import util
from ..engine.result import result_tuple
from ..sql import coercions
from ..sql import expression
from ..sql import lambdas
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import is_selectable
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import ColumnCollection
from ..sql.cache_key import HasCacheKey
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import ColumnElement
from ..sql.elements import KeyedColumnElement
from ..sql.selectable import FromClause
from ..util.langhelpers import MemoizedSlots
from ..util.typing import de_stringify_annotation as _de_stringify_annotation
from ..util.typing import (
from ..util.typing import eval_name_only as _eval_name_only
from ..util.typing import is_origin_of_cls
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import typing_get_origin
@property
def class_(self) -> Type[_O]:
    """Return the mapped class ultimately represented by this
        :class:`.AliasedInsp`."""
    return self.mapper.class_