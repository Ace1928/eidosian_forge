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
@lazy_property
def receiver_connected(self) -> Signal:
    """Emitted after each :meth:`connect`.

        The signal sender is the signal instance, and the :meth:`connect`
        arguments are passed through: *receiver*, *sender*, and *weak*.

        .. versionadded:: 1.2

        """
    return Signal(doc='Emitted after a receiver connects.')