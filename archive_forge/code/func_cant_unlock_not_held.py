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
def cant_unlock_not_held(locked_object):
    """An attempt to unlock failed because the object was not locked.

    This provides a policy point from which we can generate either a warning or
    an exception.
    """
    if 'unlock' in debug.debug_flags:
        warnings.warn('{!r} is already unlocked'.format(locked_object), stacklevel=3)
    else:
        raise errors.LockNotHeld(locked_object)