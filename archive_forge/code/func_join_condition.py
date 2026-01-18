from __future__ import annotations
from collections import deque
import copy
from itertools import chain
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import visitors
from ._typing import is_text_clause
from .annotation import _deep_annotate as _deep_annotate  # noqa: F401
from .annotation import _deep_deannotate as _deep_deannotate  # noqa: F401
from .annotation import _shallow_annotate as _shallow_annotate  # noqa: F401
from .base import _expand_cloned
from .base import _from_objects
from .cache_key import HasCacheKey as HasCacheKey  # noqa: F401
from .ddl import sort_tables as sort_tables  # noqa: F401
from .elements import _find_columns as _find_columns
from .elements import _label_reference
from .elements import _textual_label_reference
from .elements import BindParameter
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Grouping
from .elements import KeyedColumnElement
from .elements import Label
from .elements import NamedColumn
from .elements import Null
from .elements import UnaryExpression
from .schema import Column
from .selectable import Alias
from .selectable import FromClause
from .selectable import FromGrouping
from .selectable import Join
from .selectable import ScalarSelect
from .selectable import SelectBase
from .selectable import TableClause
from .visitors import _ET
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def join_condition(a: FromClause, b: FromClause, a_subset: Optional[FromClause]=None, consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]]=None) -> ColumnElement[bool]:
    """Create a join condition between two tables or selectables.

    e.g.::

        join_condition(tablea, tableb)

    would produce an expression along the lines of::

        tablea.c.id==tableb.c.tablea_id

    The join is determined based on the foreign key relationships
    between the two selectables.   If there are multiple ways
    to join, or no way to join, an error is raised.

    :param a_subset: An optional expression that is a sub-component
        of ``a``.  An attempt will be made to join to just this sub-component
        first before looking at the full ``a`` construct, and if found
        will be successful even if there are other ways to join to ``a``.
        This allows the "right side" of a join to be passed thereby
        providing a "natural join".

    """
    return Join._join_condition(a, b, a_subset=a_subset, consider_as_foreign_keys=consider_as_foreign_keys)