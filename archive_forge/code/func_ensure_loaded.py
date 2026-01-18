import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
def ensure_loaded(self):
    """Ensure we have read names from disk.

        :return: True if the disk names had not been previously read.
        """
    if not self.repo.is_locked():
        raise errors.ObjectNotLocked(self.repo)
    if self._names is None:
        self._names = {}
        self._packs_at_load = set()
        for index, key, value in self._iter_disk_pack_index():
            name = key[0].decode('ascii')
            self._names[name] = self._parse_index_sizes(value)
            self._packs_at_load.add((name, value))
        result = True
    else:
        result = False
    self.all_packs()
    return result