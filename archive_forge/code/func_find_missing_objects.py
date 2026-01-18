import sys
from dulwich.object_store import MissingObjectFinder, peel_sha
from dulwich.protocol import Protocol
from dulwich.server import (Backend, BackendRepo, ReceivePackHandler,
from .. import errors, trace
from ..controldir import ControlDir
from .mapping import decode_git_path, default_mapping
from .object_store import BazaarObjectStore, get_object_store
from .refs import get_refs_container
def find_missing_objects(self, determine_wants, graph_walker, progress, get_tagged=None):
    """Yield git objects to send to client """
    with self.object_store.lock_read():
        wants = determine_wants(self.get_refs())
        have = self.object_store.find_common_revisions(graph_walker)
        if wants is None:
            return
        shallows = getattr(graph_walker, 'shallow', frozenset())
        if isinstance(self.object_store, BazaarObjectStore):
            return self.object_store.find_missing_objects(have, wants, shallow=shallows, progress=progress, get_tagged=get_tagged, lossy=True)
        else:
            return MissingObjectFinder(self.object_store, have, wants, shallow=shallows, progress=progress)