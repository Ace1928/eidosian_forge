from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
@util.preload_module('sqlalchemy.sql.util')
def reduce_columns(self, only_synonyms: bool=True) -> Select[Any]:
    """Return a new :func:`_expression.select` construct with redundantly
        named, equivalently-valued columns removed from the columns clause.

        "Redundant" here means two columns where one refers to the
        other either based on foreign key, or via a simple equality
        comparison in the WHERE clause of the statement.   The primary purpose
        of this method is to automatically construct a select statement
        with all uniquely-named columns, without the need to use
        table-qualified labels as
        :meth:`_expression.Select.set_label_style`
        does.

        When columns are omitted based on foreign key, the referred-to
        column is the one that's kept.  When columns are omitted based on
        WHERE equivalence, the first column in the columns clause is the
        one that's kept.

        :param only_synonyms: when True, limit the removal of columns
         to those which have the same name as the equivalent.   Otherwise,
         all columns that are equivalent to another are removed.

        """
    woc: Select[Any]
    woc = self.with_only_columns(*util.preloaded.sql_util.reduce_columns(self._all_selected_columns, *self._where_criteria + self._from_obj, only_synonyms=only_synonyms))
    return woc