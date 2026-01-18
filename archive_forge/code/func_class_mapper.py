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
def class_mapper(class_: Type[_O], configure: bool=True) -> Mapper[_O]:
    """Given a class, return the primary :class:`_orm.Mapper` associated
    with the key.

    Raises :exc:`.UnmappedClassError` if no mapping is configured
    on the given class, or :exc:`.ArgumentError` if a non-class
    object is passed.

    Equivalent functionality is available via the :func:`_sa.inspect`
    function as::

        inspect(some_mapped_class)

    Using the inspection system will raise
    :class:`sqlalchemy.exc.NoInspectionAvailable` if the class is not mapped.

    """
    mapper = _inspect_mapped_class(class_, configure=configure)
    if mapper is None:
        if not isinstance(class_, type):
            raise sa_exc.ArgumentError("Class object expected, got '%r'." % (class_,))
        raise exc.UnmappedClassError(class_)
    else:
        return mapper