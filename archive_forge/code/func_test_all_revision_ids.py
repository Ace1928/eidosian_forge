import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_all_revision_ids(self):
    commit_id = self._do_commit()
    self.assertEqual([default_mapping.revision_id_foreign_to_bzr(commit_id)], self.git_repo.all_revision_ids())