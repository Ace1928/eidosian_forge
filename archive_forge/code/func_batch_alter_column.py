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
def batch_alter_column(cls, operations: BatchOperations, column_name: str, *, nullable: Optional[bool]=None, comment: Optional[Union[str, Literal[False]]]=False, server_default: Any=False, new_column_name: Optional[str]=None, type_: Optional[Union[TypeEngine[Any], Type[TypeEngine[Any]]]]=None, existing_type: Optional[Union[TypeEngine[Any], Type[TypeEngine[Any]]]]=None, existing_server_default: Optional[Union[str, bool, Identity, Computed]]=False, existing_nullable: Optional[bool]=None, existing_comment: Optional[str]=None, insert_before: Optional[str]=None, insert_after: Optional[str]=None, **kw: Any) -> None:
    """Issue an "alter column" instruction using the current
        batch migration context.

        Parameters are the same as that of :meth:`.Operations.alter_column`,
        as well as the following option(s):

        :param insert_before: String name of an existing column which this
         column should be placed before, when creating the new table.

        :param insert_after: String name of an existing column which this
         column should be placed after, when creating the new table.  If
         both :paramref:`.BatchOperations.alter_column.insert_before`
         and :paramref:`.BatchOperations.alter_column.insert_after` are
         omitted, the column is inserted after the last existing column
         in the table.

        .. seealso::

            :meth:`.Operations.alter_column`


        """
    alt = cls(operations.impl.table_name, column_name, schema=operations.impl.schema, existing_type=existing_type, existing_server_default=existing_server_default, existing_nullable=existing_nullable, existing_comment=existing_comment, modify_name=new_column_name, modify_type=type_, modify_server_default=server_default, modify_nullable=nullable, modify_comment=comment, insert_before=insert_before, insert_after=insert_after, **kw)
    return operations.invoke(alt)