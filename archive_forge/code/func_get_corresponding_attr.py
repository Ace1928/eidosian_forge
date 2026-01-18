from __future__ import annotations
from typing import Type
from . import exc as orm_exc
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .. import exc
from .. import inspect
from ..sql import and_
from ..sql import operators
from ..sql.sqltypes import Integer
from ..sql.sqltypes import Numeric
from ..util import warn_deprecated
def get_corresponding_attr(obj):
    if obj is None:
        return _NO_OBJECT
    state = inspect(obj)
    dict_ = state.dict
    value = impl.get(state, dict_, passive=PassiveFlag.PASSIVE_NO_FETCH)
    if value is LoaderCallableStatus.PASSIVE_NO_RESULT:
        return _EXPIRED_OBJECT
    return value