import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_has_revision(self):
    GitRepo.init(self.test_dir)
    commit_id = self._do_commit()
    repo = Repository.open('.')
    self.assertFalse(repo.has_revision(b'foobar'))
    revid = default_mapping.revision_id_foreign_to_bzr(commit_id)
    self.assertTrue(repo.has_revision(revid))