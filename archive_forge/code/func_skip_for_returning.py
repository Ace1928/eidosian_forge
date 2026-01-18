from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
def skip_for_returning(orm_context: ORMExecuteState) -> Any:
    bind = orm_context.session.get_bind(**orm_context.bind_arguments)
    nonlocal can_use_returning
    per_bind_result = cls.can_use_returning(bind.dialect, mapper, is_update_from=update_options._is_update_from, is_delete_using=update_options._is_delete_using, is_executemany=orm_context.is_executemany)
    if can_use_returning is not None:
        if can_use_returning != per_bind_result:
            raise sa_exc.InvalidRequestError("For synchronize_session='fetch', can't mix multiple backends where some support RETURNING and others don't")
    elif orm_context.is_executemany and (not per_bind_result):
        raise sa_exc.InvalidRequestError("For synchronize_session='fetch', can't use multiple parameter sets in ORM mode, which this backend does not support with RETURNING")
    else:
        can_use_returning = per_bind_result
    if per_bind_result:
        return _result.null_result()
    else:
        return None