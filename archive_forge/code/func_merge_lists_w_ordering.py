from __future__ import annotations
import operator
import threading
import types
import typing
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
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import ValuesView
import weakref
from ._has_cy import HAS_CYEXTENSION
from .typing import is_non_string_iterable
from .typing import Literal
from .typing import Protocol
def merge_lists_w_ordering(a: List[Any], b: List[Any]) -> List[Any]:
    """merge two lists, maintaining ordering as much as possible.

    this is to reconcile vars(cls) with cls.__annotations__.

    Example::

        >>> a = ['__tablename__', 'id', 'x', 'created_at']
        >>> b = ['id', 'name', 'data', 'y', 'created_at']
        >>> merge_lists_w_ordering(a, b)
        ['__tablename__', 'id', 'name', 'data', 'y', 'x', 'created_at']

    This is not necessarily the ordering that things had on the class,
    in this case the class is::

        class User(Base):
            __tablename__ = "users"

            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str]
            data: Mapped[Optional[str]]
            x = Column(Integer)
            y: Mapped[int]
            created_at: Mapped[datetime.datetime] = mapped_column()

    But things are *mostly* ordered.

    The algorithm could also be done by creating a partial ordering for
    all items in both lists and then using topological_sort(), but that
    is too much overhead.

    Background on how I came up with this is at:
    https://gist.github.com/zzzeek/89de958cf0803d148e74861bd682ebae

    """
    overlap = set(a).intersection(b)
    result = []
    current, other = (iter(a), iter(b))
    while True:
        for element in current:
            if element in overlap:
                overlap.discard(element)
                other, current = (current, other)
                break
            result.append(element)
        else:
            result.extend(other)
            break
    return result