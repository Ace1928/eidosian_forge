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
def batch_create_table_comment(cls, operations: BatchOperations, comment: Optional[str], *, existing_comment: Optional[str]=None) -> None:
    """Emit a COMMENT ON operation to set the comment for a table
        using the current batch migration context.

        :param comment: string value of the comment being registered against
         the specified table.
        :param existing_comment: String value of a comment
         already registered on the specified table, used within autogenerate
         so that the operation is reversible, but not required for direct
         use.

        """
    op = cls(operations.impl.table_name, comment, existing_comment=existing_comment, schema=operations.impl.schema)
    return operations.invoke(op)