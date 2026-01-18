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
def missing_parents_chain(self, search, sources):
    """Chain multiple streams together to handle stacking.

        :param search: The overall search to satisfy with streams.
        :param sources: A list of Repository objects to query.
        """
    self.from_serialiser = self.from_repository._format._serializer
    self.seen_revs = set()
    self.referenced_revs = set()
    while not search.is_empty() and len(sources) > 1:
        source = sources.pop(0)
        stream = self._get_stream(source, search)
        for kind, substream in stream:
            if kind != 'revisions':
                yield (kind, substream)
            else:
                yield (kind, self.missing_parents_rev_handler(substream))
        search = search.refine(self.seen_revs, self.referenced_revs)
        self.seen_revs = set()
        self.referenced_revs = set()
    if not search.is_empty():
        for kind, stream in self._get_stream(sources[0], search):
            yield (kind, stream)