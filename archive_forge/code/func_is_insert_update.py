from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from .. import exc
from .. import util
from ..inspection import Inspectable
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypeAlias
def is_insert_update(c: ClauseElement) -> TypeGuard[ValuesBase]:
    return c.is_dml and (c.is_insert or c.is_update)