from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
def get_current_parameters(self, isolate_multiinsert_groups=True):
    """Return a dictionary of parameters applied to the current row.

        This method can only be used in the context of a user-defined default
        generation function, e.g. as described at
        :ref:`context_default_functions`. When invoked, a dictionary is
        returned which includes entries for each column/value pair that is part
        of the INSERT or UPDATE statement. The keys of the dictionary will be
        the key value of each :class:`_schema.Column`,
        which is usually synonymous
        with the name.

        :param isolate_multiinsert_groups=True: indicates that multi-valued
         INSERT constructs created using :meth:`_expression.Insert.values`
         should be
         handled by returning only the subset of parameters that are local
         to the current column default invocation.   When ``False``, the
         raw parameters of the statement are returned including the
         naming convention used in the case of multi-valued INSERT.

        .. versionadded:: 1.2  added
           :meth:`.DefaultExecutionContext.get_current_parameters`
           which provides more functionality over the existing
           :attr:`.DefaultExecutionContext.current_parameters`
           attribute.

        .. seealso::

            :attr:`.DefaultExecutionContext.current_parameters`

            :ref:`context_default_functions`

        """
    try:
        parameters = self.current_parameters
        column = self.current_column
    except AttributeError:
        raise exc.InvalidRequestError('get_current_parameters() can only be invoked in the context of a Python side column default function')
    else:
        assert column is not None
        assert parameters is not None
    compile_state = cast('DMLState', cast(SQLCompiler, self.compiled).compile_state)
    assert compile_state is not None
    if isolate_multiinsert_groups and dml.isinsert(compile_state) and compile_state._has_multi_parameters:
        if column._is_multiparam_column:
            index = column.index + 1
            d = {column.original.key: parameters[column.key]}
        else:
            d = {column.key: parameters[column.key]}
            index = 0
        assert compile_state._dict_parameters is not None
        keys = compile_state._dict_parameters.keys()
        d.update(((key, parameters['%s_m%d' % (key, index)]) for key in keys))
        return d
    else:
        return parameters