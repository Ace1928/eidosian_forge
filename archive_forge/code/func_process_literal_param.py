from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def process_literal_param(self, value: Optional[_T], dialect: Dialect) -> str:
    """Receive a literal parameter value to be rendered inline within
        a statement.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. Unlike other SQL
            compilation methods, it is passed a specific Python value to be
            rendered as a string. However it should not be confused with the
            :meth:`_types.TypeDecorator.process_bind_param` method, which is
            the more typical method that processes the actual value passed to a
            particular parameter at statement execution time.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for incoming data values
        that are in the special case of being rendered as literals.

        The returned string will be rendered into the output string.

        """
    raise NotImplementedError()