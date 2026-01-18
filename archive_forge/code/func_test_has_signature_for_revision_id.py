import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_has_signature_for_revision_id(self):
    self.assertEqual(False, self.git_repo.has_signature_for_revision_id(revision.NULL_REVISION))