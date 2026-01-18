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
def all_packs(self):
    """Return a list of all the Pack objects this repository has.

        Note that an in-progress pack being created is not returned.

        :return: A list of Pack objects for all the packs in the repository.
        """
    result = []
    for name in self.names():
        result.append(self.get_pack_by_name(name))
    return result