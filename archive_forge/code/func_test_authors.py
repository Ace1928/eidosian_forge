import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_authors(self):
    wt = self.make_branch_and_tree('.', format='git')
    revid = wt.commit('base', allow_pointless=True, revprops={'authors': 'Jelmer Vernooij <jelmer@example.com>\nMartin Packman <bz2@example.com>\n'})
    rev = wt.branch.repository.get_revision(revid)
    r = dulwich.repo.Repo('.')
    self.assertEqual(r[r.head()].author, b'Jelmer Vernooij <jelmer@example.com>')
    self.assertEqual(b'base\n\nCo-authored-by: Martin Packman <bz2@example.com>\n', r[r.head()].message)