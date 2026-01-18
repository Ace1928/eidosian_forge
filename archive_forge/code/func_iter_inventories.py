import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
def iter_inventories(self, revision_ids, ordering=None):
    """Get many inventories by revision_ids.

        This will buffer some or all of the texts used in constructing the
        inventories in memory, but will only parse a single inventory at a
        time.

        :param revision_ids: The expected revision ids of the inventories.
        :param ordering: optional ordering, e.g. 'topological'.  If not
            specified, the order of revision_ids will be preserved (by
            buffering if necessary).
        :return: An iterator of inventories.
        """
    if None in revision_ids or _mod_revision.NULL_REVISION in revision_ids:
        raise ValueError('cannot get null revision inventory')
    for inv, revid in self._iter_inventories(revision_ids, ordering):
        if inv is None:
            raise errors.NoSuchRevision(self, revid)
        yield inv