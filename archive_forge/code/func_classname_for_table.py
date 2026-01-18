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
def classname_for_table(base: Type[Any], tablename: str, table: Table) -> str:
    """Return the class name that should be used, given the name
    of a table.

    The default implementation is::

        return str(tablename)

    Alternate implementations can be specified using the
    :paramref:`.AutomapBase.prepare.classname_for_table`
    parameter.

    :param base: the :class:`.AutomapBase` class doing the prepare.

    :param tablename: string name of the :class:`_schema.Table`.

    :param table: the :class:`_schema.Table` object itself.

    :return: a string class name.

     .. note::

        In Python 2, the string used for the class name **must** be a
        non-Unicode object, e.g. a ``str()`` object.  The ``.name`` attribute
        of :class:`_schema.Table` is typically a Python unicode subclass,
        so the
        ``str()`` function should be applied to this name, after accounting for
        any non-ASCII characters.

    """
    return str(tablename)