from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
@classmethod
def batch_drop_index(cls, operations: BatchOperations, index_name: str, **kw: Any) -> None:
    """Issue a "drop index" instruction using the
        current batch migration context.

        .. seealso::

            :meth:`.Operations.drop_index`

        """
    op = cls(index_name, table_name=operations.impl.table_name, schema=operations.impl.schema, **kw)
    return operations.invoke(op)