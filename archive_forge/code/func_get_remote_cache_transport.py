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
def get_remote_cache_transport(repository):
    """Retrieve the transport to use when accessing (unwritable) remote
    repositories.
    """
    uuid = getattr(repository, 'uuid', None)
    if uuid is None:
        path = get_cache_dir()
    else:
        path = os.path.join(get_cache_dir(), uuid)
        if not os.path.isdir(path):
            os.mkdir(path)
    return get_transport_from_path(path)