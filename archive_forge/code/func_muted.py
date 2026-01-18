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
@contextmanager
def muted(self) -> t.Generator[None, None, None]:
    """Context manager for temporarily disabling signal.
        Useful for test purposes.
        """
    self.is_muted = True
    try:
        yield None
    except Exception as e:
        raise e
    finally:
        self.is_muted = False