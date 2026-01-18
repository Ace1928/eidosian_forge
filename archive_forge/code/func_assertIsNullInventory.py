import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def assertIsNullInventory(self, inv):
    self.assertEqual(inv.root, None)
    self.assertEqual(inv.revision_id, revision.NULL_REVISION)
    self.assertEqual(list(inv.iter_entries()), [])