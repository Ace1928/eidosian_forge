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
def find_left_clause_to_join_from(clauses: Sequence[FromClause], join_to: _JoinTargetElement, onclause: Optional[ColumnElement[Any]]) -> List[int]:
    """Given a list of FROM clauses, a selectable,
    and optional ON clause, return a list of integer indexes from the
    clauses list indicating the clauses that can be joined from.

    The presence of an "onclause" indicates that at least one clause can
    definitely be joined from; if the list of clauses is of length one
    and the onclause is given, returns that index.   If the list of clauses
    is more than length one, and the onclause is given, attempts to locate
    which clauses contain the same columns.

    """
    idx = []
    selectables = set(_from_objects(join_to))
    if len(clauses) > 1 and onclause is not None:
        resolve_ambiguity = True
        cols_in_onclause = _find_columns(onclause)
    else:
        resolve_ambiguity = False
        cols_in_onclause = None
    for i, f in enumerate(clauses):
        for s in selectables.difference([f]):
            if resolve_ambiguity:
                assert cols_in_onclause is not None
                if set(f.c).union(s.c).issuperset(cols_in_onclause):
                    idx.append(i)
                    break
            elif onclause is not None or Join._can_join(f, s):
                idx.append(i)
                break
    if len(idx) > 1:
        toremove = set(chain(*[_expand_cloned(f._hide_froms) for f in clauses]))
        idx = [i for i in idx if clauses[i] not in toremove]
    if not idx and onclause is not None:
        return list(range(len(clauses)))
    else:
        return idx