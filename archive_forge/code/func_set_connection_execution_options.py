from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
def set_connection_execution_options(self, connection: Connection, opts: CoreExecuteOptionsParameter) -> None:
    """Establish execution options for a given connection.

        This is implemented by :class:`.DefaultDialect` in order to implement
        the :paramref:`_engine.Connection.execution_options.isolation_level`
        execution option.  Dialects can intercept various execution options
        which may need to modify state on a particular DBAPI connection.

        .. versionadded:: 1.4

        """
    raise NotImplementedError()