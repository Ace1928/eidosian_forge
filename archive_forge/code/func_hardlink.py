import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def hardlink(self, source, link_name):
    """Create a hardlink pointing to source named link_name."""
    raise errors.TransportNotPossible('Hard links are not supported on %s' % self)