import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def listable(self):
    """Return True if this store is able to be listed."""
    return self._transport.listable()