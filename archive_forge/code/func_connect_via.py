from __future__ import annotations
import typing as t
from collections import defaultdict
from contextlib import contextmanager
from inspect import iscoroutinefunction
from warnings import warn
from weakref import WeakValueDictionary
from blinker._utilities import annotatable_weakref
from blinker._utilities import hashable_identity
from blinker._utilities import IdentityType
from blinker._utilities import lazy_property
from blinker._utilities import reference
from blinker._utilities import symbol
from blinker._utilities import WeakTypes
def connect_via(self, sender: t.Any, weak: bool=False) -> t.Callable[[T_callable], T_callable]:
    """Connect the decorated function as a receiver for *sender*.

        :param sender: Any object or :obj:`ANY`.  The decorated function
          will only receive :meth:`send` emissions sent by *sender*.  If
          ``ANY``, the receiver will always be notified.  A function may be
          decorated multiple times with differing *sender* values.

        :param weak: If true, the Signal will hold a weakref to the
          decorated function and automatically disconnect when *receiver*
          goes out of scope or is garbage collected.  Unlike
          :meth:`connect`, this defaults to False.

        The decorated function will be invoked by :meth:`send` with
          `sender=` as a single positional argument and any ``kwargs`` that
          were provided to the call to :meth:`send`.


        .. versionadded:: 1.1

        """

    def decorator(fn: T_callable) -> T_callable:
        self.connect(fn, sender, weak)
        return fn
    return decorator