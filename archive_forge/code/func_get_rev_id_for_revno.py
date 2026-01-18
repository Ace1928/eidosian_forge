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
def get_rev_id_for_revno(self, revno, known_pair):
    """See Repository.get_rev_id_for_revno."""
    path = self.controldir._path_for_remote_call(self._client)
    try:
        if self._client._medium._is_remote_before((1, 17)):
            return self._get_rev_id_for_revno_vfs(revno, known_pair)
        response = self._call(b'Repository.get_rev_id_for_revno', path, revno, known_pair)
    except errors.UnknownSmartMethod:
        self._client._medium._remember_remote_is_before((1, 17))
        return self._get_rev_id_for_revno_vfs(revno, known_pair)
    except UnknownErrorFromSmartServer as e:
        if len(e.error_tuple) < 3:
            raise
        if e.error_tuple[:2] != (b'error', b'ValueError'):
            raise
        m = re.match(b'requested revno \\(([0-9]+)\\) is later than given known revno \\(([0-9]+)\\)', e.error_tuple[2])
        if not m:
            raise
        raise errors.RevnoOutOfBounds(int(m.group(1)), (0, int(m.group(2))))
    if response[0] == b'ok':
        return (True, response[1])
    elif response[0] == b'history-incomplete':
        known_pair = response[1:3]
        for fallback in self._fallback_repositories:
            found, result = fallback.get_rev_id_for_revno(revno, known_pair)
            if found:
                return (True, result)
            else:
                known_pair = result
        return (False, known_pair)
    else:
        raise errors.UnexpectedSmartServerResponse(response)