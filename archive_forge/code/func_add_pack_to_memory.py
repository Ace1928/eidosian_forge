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
def add_pack_to_memory(self, pack):
    """Make a Pack object available to the repository to satisfy queries.

        :param pack: A Pack object.
        """
    if pack.name in self._packs_by_name:
        raise AssertionError('pack {} already in _packs_by_name'.format(pack.name))
    self.packs.append(pack)
    self._packs_by_name[pack.name] = pack
    self.revision_index.add_index(pack.revision_index, pack)
    self.inventory_index.add_index(pack.inventory_index, pack)
    self.text_index.add_index(pack.text_index, pack)
    self.signature_index.add_index(pack.signature_index, pack)
    if self.chk_index is not None:
        self.chk_index.add_index(pack.chk_index, pack)