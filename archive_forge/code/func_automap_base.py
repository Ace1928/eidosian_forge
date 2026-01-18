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
def automap_base(declarative_base: Optional[Type[Any]]=None, **kw: Any) -> Any:
    """Produce a declarative automap base.

    This function produces a new base class that is a product of the
    :class:`.AutomapBase` class as well a declarative base produced by
    :func:`.declarative.declarative_base`.

    All parameters other than ``declarative_base`` are keyword arguments
    that are passed directly to the :func:`.declarative.declarative_base`
    function.

    :param declarative_base: an existing class produced by
     :func:`.declarative.declarative_base`.  When this is passed, the function
     no longer invokes :func:`.declarative.declarative_base` itself, and all
     other keyword arguments are ignored.

    :param \\**kw: keyword arguments are passed along to
     :func:`.declarative.declarative_base`.

    """
    if declarative_base is None:
        Base = _declarative_base(**kw)
    else:
        Base = declarative_base
    return type(Base.__name__, (AutomapBase, Base), {'__abstract__': True, 'classes': util.Properties({}), 'by_module': util.Properties({}), '_sa_automapbase_bookkeeping': _Bookkeeping(set())})