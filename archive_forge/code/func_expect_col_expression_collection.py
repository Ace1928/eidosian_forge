from __future__ import annotations
import collections.abc as collections_abc
import numbers
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import operators
from . import roles
from . import visitors
from ._typing import is_from_clause
from .base import ExecutableOption
from .base import Options
from .cache_key import HasCacheKey
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def expect_col_expression_collection(role: Type[roles.DDLConstraintColumnRole], expressions: Iterable[_DDLColumnArgument]) -> Iterator[Tuple[Union[str, Column[Any]], Optional[ColumnClause[Any]], Optional[str], Optional[Union[Column[Any], str]]]]:
    for expr in expressions:
        strname = None
        column = None
        resolved: Union[Column[Any], str] = expect(role, expr)
        if isinstance(resolved, str):
            assert isinstance(expr, str)
            strname = resolved = expr
        else:
            cols: List[Column[Any]] = []
            col_append: _TraverseCallableType[Column[Any]] = cols.append
            visitors.traverse(resolved, {}, {'column': col_append})
            if cols:
                column = cols[0]
        add_element = column if column is not None else strname
        yield (resolved, column, strname, add_element)