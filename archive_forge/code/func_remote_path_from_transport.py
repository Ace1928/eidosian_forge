from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def remote_path_from_transport(self, transport):
    """Convert transport into a path suitable for using in a request.

        Note that the resulting remote path doesn't encode the host name or
        anything but path, so it is only safe to use it in requests sent over
        the medium from the matching transport.
        """
    return self._medium.remote_path_from_transport(transport).encode('utf-8')