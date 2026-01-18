import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_has_git_repo(self):
    GitRepo.init(self.test_dir)
    repo = Repository.open('.')
    self.assertIsInstance(repo._git, dulwich.repo.BaseRepo)