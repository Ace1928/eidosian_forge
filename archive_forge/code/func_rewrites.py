from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from .. import util
from ..operations import ops
def rewrites(self, operator: Union[Type[AddColumnOp], Type[MigrateOperation], Type[AlterColumnOp], Type[CreateTableOp], Type[ModifyTableOps]]) -> Callable[..., Any]:
    """Register a function as rewriter for a given type.

        The function should receive three arguments, which are
        the :class:`.MigrationContext`, a ``revision`` tuple, and
        an op directive of the type indicated.  E.g.::

            @writer1.rewrites(ops.AddColumnOp)
            def add_column_nullable(context, revision, op):
                op.column.nullable = True
                return op

        """
    return self.dispatch.dispatch_for(operator)