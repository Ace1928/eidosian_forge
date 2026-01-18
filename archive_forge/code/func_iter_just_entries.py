from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def iter_just_entries(self):
    """Iterate over all entries.

        Unlike iter_entries(), just the entries are returned (not (path, ie))
        and the order of entries is undefined.

        XXX: We may not want to merge this into bzr.dev.
        """
    for key, entry in self.id_to_entry.iteritems():
        file_id = key[0]
        ie = self._fileid_to_entry_cache.get(file_id, None)
        if ie is None:
            ie = self._bytes_to_entry(entry)
            self._fileid_to_entry_cache[file_id] = ie
        yield ie