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
def lookup_tree_id(self, fileid, revision):
    row = self.db.execute('select sha1 from trees where fileid = ? and revid = ?', (fileid, revision)).fetchone()
    if row is not None:
        return row[0]
    raise KeyError(fileid)