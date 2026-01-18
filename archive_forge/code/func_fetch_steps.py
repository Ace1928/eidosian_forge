from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def fetch_steps(self, br_a, br_b, writable_a):
    """A foreign test method for testing fetch locally and remotely."""
    repo_b = br_b.repository
    self.assertFalse(repo_b.has_revision(revision_history(br_a)[3]))
    self.assertTrue(repo_b.has_revision(revision_history(br_a)[2]))
    self.assertEqual(len(revision_history(br_b)), 7)
    br_b.fetch(br_a, revision_history(br_a)[2])
    self.assertEqual(len(revision_history(br_b)), 7)
    self.assertFalse(repo_b.has_revision(revision_history(br_a)[3]))
    br_b.fetch(br_a, revision_history(br_a)[3])
    self.assertTrue(repo_b.has_revision(revision_history(br_a)[3]))
    self.assertFalse(has_revision(br_a, revision_history(br_b)[6]))
    self.assertTrue(br_a.repository.has_revision(revision_history(br_b)[5]))
    br_b4 = self.make_branch('br_4')
    br_b4.fetch(br_b)
    writable_a.fetch(br_b)
    self.assertTrue(has_revision(br_a, revision_history(br_b)[3]))
    self.assertTrue(has_revision(br_a, revision_history(br_b)[4]))
    br_b2 = self.make_branch('br_b2')
    br_b2.fetch(br_b)
    self.assertTrue(has_revision(br_b2, revision_history(br_b)[4]))
    self.assertTrue(has_revision(br_b2, revision_history(br_a)[2]))
    self.assertFalse(has_revision(br_b2, revision_history(br_a)[3]))
    br_a2 = self.make_branch('br_a2')
    br_a2.fetch(br_a)
    self.assertTrue(has_revision(br_a2, revision_history(br_b)[4]))
    self.assertTrue(has_revision(br_a2, revision_history(br_a)[3]))
    self.assertTrue(has_revision(br_a2, revision_history(br_a)[2]))
    br_a3 = self.make_branch('br_a3')
    br_a3.fetch(br_a2)
    for revno in range(4):
        self.assertFalse(br_a3.repository.has_revision(revision_history(br_a)[revno]))
    br_a3.fetch(br_a2, revision_history(br_a)[2])
    br_a3.fetch(br_a2, revision_history(br_a)[3])
    self.assertRaises(errors.NoSuchRevision, br_a3.fetch, br_a2, 'pizza')