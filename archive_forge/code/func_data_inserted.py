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
def data_inserted(self):
    """True if data has been added to this pack."""
    return bool(self.get_revision_count() or self.inventory_index.key_count() or self.text_index.key_count() or self.signature_index.key_count() or (self.chk_index is not None and self.chk_index.key_count()))