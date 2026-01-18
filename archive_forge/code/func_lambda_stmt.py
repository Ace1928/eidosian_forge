from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def lambda_stmt(lmb: _StmtLambdaType, enable_tracking: bool=True, track_closure_variables: bool=True, track_on: Optional[object]=None, global_track_bound_values: bool=True, track_bound_values: bool=True, lambda_cache: Optional[_LambdaCacheType]=None) -> StatementLambdaElement:
    """Produce a SQL statement that is cached as a lambda.

    The Python code object within the lambda is scanned for both Python
    literals that will become bound parameters as well as closure variables
    that refer to Core or ORM constructs that may vary.   The lambda itself
    will be invoked only once per particular set of constructs detected.

    E.g.::

        from sqlalchemy import lambda_stmt

        stmt = lambda_stmt(lambda: table.select())
        stmt += lambda s: s.where(table.c.id == 5)

        result = connection.execute(stmt)

    The object returned is an instance of :class:`_sql.StatementLambdaElement`.

    .. versionadded:: 1.4

    :param lmb: a Python function, typically a lambda, which takes no arguments
     and returns a SQL expression construct
    :param enable_tracking: when False, all scanning of the given lambda for
     changes in closure variables or bound parameters is disabled.  Use for
     a lambda that produces the identical results in all cases with no
     parameterization.
    :param track_closure_variables: when False, changes in closure variables
     within the lambda will not be scanned.   Use for a lambda where the
     state of its closure variables will never change the SQL structure
     returned by the lambda.
    :param track_bound_values: when False, bound parameter tracking will
     be disabled for the given lambda.  Use for a lambda that either does
     not produce any bound values, or where the initial bound values never
     change.
    :param global_track_bound_values: when False, bound parameter tracking
     will be disabled for the entire statement including additional links
     added via the :meth:`_sql.StatementLambdaElement.add_criteria` method.
    :param lambda_cache: a dictionary or other mapping-like object where
     information about the lambda's Python code as well as the tracked closure
     variables in the lambda itself will be stored.   Defaults
     to a global LRU cache.  This cache is independent of the "compiled_cache"
     used by the :class:`_engine.Connection` object.

    .. seealso::

        :ref:`engine_lambda_caching`


    """
    return StatementLambdaElement(lmb, roles.StatementRole, LambdaOptions(enable_tracking=enable_tracking, track_on=track_on, track_closure_variables=track_closure_variables, global_track_bound_values=global_track_bound_values, track_bound_values=track_bound_values, lambda_cache=lambda_cache))