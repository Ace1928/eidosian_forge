from __future__ import annotations
from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self
def filterrows(make_row: Optional[Callable[..., _R]], rows: List[Any], strategy: Optional[Callable[[List[Any]], Any]], uniques: Set[Any]) -> List[_R]:
    if make_row:
        rows = [make_row(row) for row in rows]
    if strategy:
        made_rows = ((made_row, strategy(made_row)) for made_row in rows)
    else:
        made_rows = ((made_row, made_row) for made_row in rows)
    return [made_row for made_row, sig_row in made_rows if sig_row not in uniques and (not uniques.add(sig_row))]