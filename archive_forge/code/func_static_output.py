from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.schema import FetchedValue
from typing_extensions import Literal
from .migration import _ProxyTransaction
from .migration import MigrationContext
from .. import util
from ..operations import Operations
from ..script.revision import _GetRevArg
def static_output(self, text: str) -> None:
    """Emit text directly to the "offline" SQL stream.

        Typically this is for emitting comments that
        start with --.  The statement is not treated
        as a SQL execution, no ; or batch separator
        is added, etc.

        """
    self.get_context().impl.static_output(text)