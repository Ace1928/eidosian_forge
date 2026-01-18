import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def readlink(self, relpath):
    """Return a string representing the path to which the symbolic link points."""
    raise errors.TransportNotPossible('Dereferencing symlinks is not supported on %s' % self)