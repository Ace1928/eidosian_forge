import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_get_parent_map_null(self):
    self.assertEqual({revision.NULL_REVISION: ()}, self.git_repo.get_parent_map([revision.NULL_REVISION]))