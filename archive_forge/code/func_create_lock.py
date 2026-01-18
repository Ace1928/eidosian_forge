from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
def create_lock(self) -> None:
    """Create the lock.

        This should normally be called only when the LockableFiles directory
        is first created on disk.
        """
    self._lock.create(mode=self._dir_mode)