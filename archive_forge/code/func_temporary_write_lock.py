import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
def temporary_write_lock(self):
    """Try to grab a write lock on the file.

            On platforms that support it, this will upgrade to a write lock
            without unlocking the file.
            Otherwise, this will release the read lock, and try to acquire a
            write lock.

            :return: A token which can be used to switch back to a read lock.
            """
    self.unlock()
    try:
        wlock = _ctypes_WriteLock(self.filename)
    except errors.LockError:
        return (False, _ctypes_ReadLock(self.filename))
    return (True, wlock)