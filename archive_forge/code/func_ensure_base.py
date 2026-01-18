import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def ensure_base(self, mode=None):
    """Ensure that the directory this transport references exists.

        This will create a directory if it doesn't exist.
        :return: True if the directory was created, False otherwise.
        """
    try:
        self.mkdir('.', mode=mode)
    except (FileExists, errors.PermissionDenied):
        return False
    except errors.TransportNotPossible:
        if self.has('.'):
            return False
        raise
    else:
        return True