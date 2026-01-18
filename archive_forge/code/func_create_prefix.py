import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def create_prefix(self, mode=None):
    """Create all the directories leading down to self.base."""
    cur_transport = self
    needed = [cur_transport]
    while True:
        new_transport = cur_transport.clone('..')
        if new_transport.base == cur_transport.base:
            raise errors.CommandError('Failed to create path prefix for %s.' % cur_transport.base)
        try:
            new_transport.mkdir('.', mode=mode)
        except NoSuchFile:
            needed.append(new_transport)
            cur_transport = new_transport
        except FileExists:
            break
        else:
            break
    while needed:
        cur_transport = needed.pop()
        cur_transport.ensure_base(mode=mode)