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
def get_updater(self, rev):
    """Update an object that implements the CacheUpdater interface for
        updating this cache.
        """
    return self._cache_updater_klass(self, rev)