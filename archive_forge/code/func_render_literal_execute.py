from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
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
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
def render_literal_execute(self) -> BindParameter[_T]:
    """Produce a copy of this bound parameter that will enable the
        :paramref:`_sql.BindParameter.literal_execute` flag.

        The :paramref:`_sql.BindParameter.literal_execute` flag will
        have the effect of the parameter rendered in the compiled SQL
        string using ``[POSTCOMPILE]`` form, which is a special form that
        is converted to be a rendering of the literal value of the parameter
        at SQL execution time.    The rationale is to support caching
        of SQL statement strings that can embed per-statement literal values,
        such as LIMIT and OFFSET parameters, in the final SQL string that
        is passed to the DBAPI.   Dialects in particular may want to use
        this method within custom compilation schemes.

        .. versionadded:: 1.4.5

        .. seealso::

            :ref:`engine_thirdparty_caching`

        """
    c = ClauseElement._clone(self)
    c.literal_execute = True
    return c