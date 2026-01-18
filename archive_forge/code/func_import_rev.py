import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def import_rev(self, revid, parent_lookup=None):
    store, store_iter = self.object_iter()
    store._cache.idmap.start_write_group()
    try:
        return store_iter.import_revision(revid, lossy=True)
    except:
        store._cache.idmap.abort_write_group()
        raise
    else:
        store._cache.idmap.commit_write_group()