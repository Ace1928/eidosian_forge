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
def missing_revisions(self, revids):
    """Return set of all the revisions that are not present."""
    missing_revids = set(revids)
    for _, key, value in self._index.iter_entries(((b'commit', revid, b'X') for revid in revids)):
        missing_revids.remove(key[1])
    return missing_revids