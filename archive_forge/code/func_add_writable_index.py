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
def add_writable_index(self, index, pack):
    """Add an index which is able to have data added to it.

        There can be at most one writable index at any time.  Any
        modifications made to the knit are put into this index.

        :param index: An index from the pack parameter.
        :param pack: A Pack instance.
        """
    if self.add_callback is not None:
        raise AssertionError('%s already has a writable index through %s' % (self, self.add_callback))
    self.add_index(index, pack)
    self.data_access.set_writer(pack._writer, index, pack.access_tuple())
    self.add_callback = index.add_nodes