from ``Engineer`` to ``Employee``, we need to set up both the relationship
from __future__ import annotations
import dataclasses
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import backref
from ..orm import declarative_base as _declarative_base
from ..orm import exc as orm_exc
from ..orm import interfaces
from ..orm import relationship
from ..orm.decl_base import _DeferredMapperConfig
from ..orm.mapper import _CONFIGURE_MUTEX
from ..schema import ForeignKeyConstraint
from ..sql import and_
from ..util import Properties
from ..util.typing import Protocol
def name_for_scalar_relationship(base: Type[Any], local_cls: Type[Any], referred_cls: Type[Any], constraint: ForeignKeyConstraint) -> str:
    """Return the attribute name that should be used to refer from one
    class to another, for a scalar object reference.

    The default implementation is::

        return referred_cls.__name__.lower()

    Alternate implementations can be specified using the
    :paramref:`.AutomapBase.prepare.name_for_scalar_relationship`
    parameter.

    :param base: the :class:`.AutomapBase` class doing the prepare.

    :param local_cls: the class to be mapped on the local side.

    :param referred_cls: the class to be mapped on the referring side.

    :param constraint: the :class:`_schema.ForeignKeyConstraint` that is being
     inspected to produce this relationship.

    """
    return referred_cls.__name__.lower()