import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
@classmethod
def from_transport(self, transport):
    """Open a cache file present on a transport, or initialize one.

        :param transport: Transport to use
        :return: A BzrGitCache instance
        """
    try:
        format_name = transport.get_bytes('format')
        format = formats.get(format_name)
    except NoSuchFile:
        format = formats.get('default')
        format.initialize(transport)
    return format.open(transport)