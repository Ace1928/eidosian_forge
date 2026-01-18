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
def batch_create_check_constraint(cls, operations: BatchOperations, constraint_name: str, condition: Union[str, ColumnElement[bool], TextClause], **kw: Any) -> None:
    """Issue a "create check constraint" instruction using the
        current batch migration context.

        The batch form of this call omits the ``source`` and ``schema``
        arguments from the call.

        .. seealso::

            :meth:`.Operations.create_check_constraint`

        """
    op = cls(constraint_name, operations.impl.table_name, condition, schema=operations.impl.schema, **kw)
    return operations.invoke(op)