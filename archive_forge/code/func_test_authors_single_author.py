import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_authors_single_author(self):
    wt = self.make_branch_and_tree('.', format='git')
    revid = wt.commit('base', allow_pointless=True, revprops={'authors': 'Joe Example <joe@example.com>'})
    rev = wt.branch.repository.get_revision(revid)
    r = dulwich.repo.Repo('.')
    self.assertEqual(b'Joe Example <joe@example.com>', r[r.head()].author)