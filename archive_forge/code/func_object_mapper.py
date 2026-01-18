from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
def object_mapper(instance: _T) -> Mapper[_T]:
    """Given an object, return the primary Mapper associated with the object
    instance.

    Raises :class:`sqlalchemy.orm.exc.UnmappedInstanceError`
    if no mapping is configured.

    This function is available via the inspection system as::

        inspect(instance).mapper

    Using the inspection system will raise
    :class:`sqlalchemy.exc.NoInspectionAvailable` if the instance is
    not part of a mapping.

    """
    return object_state(instance).mapper