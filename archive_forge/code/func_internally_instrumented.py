from __future__ import annotations
import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol
@staticmethod
def internally_instrumented(fn):
    """Tag the method as instrumented.

        This tag will prevent any decoration from being applied to the
        method. Use this if you are orchestrating your own calls to
        :func:`.collection_adapter` in one of the basic SQLAlchemy
        interface methods, or to prevent an automatic ABC method
        decoration from wrapping your implementation::

            # normally an 'extend' method on a list-like class would be
            # automatically intercepted and re-implemented in terms of
            # SQLAlchemy events and append().  your implementation will
            # never be called, unless:
            @collection.internally_instrumented
            def extend(self, items): ...

        """
    fn._sa_instrumented = True
    return fn