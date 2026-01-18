from breezy import errors
from breezy.repository import WriteGroup
from breezy.tests.per_repository_reference import \
def test_add_revision_goes_to_repo(self):
    tree = self.make_branch_and_tree('sample')
    revid = tree.commit('one')
    inv = tree.branch.repository.get_inventory(revid)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    rev = tree.branch.repository.get_revision(revid)
    base = self.make_repository('base')
    repo = self.make_referring('referring', base)
    with repo.lock_write(), WriteGroup(repo):
        rev = tree.branch.repository.get_revision(revid)
        repo.texts.add_lines((inv.root.file_id, revid), [], [])
        repo.add_revision(revid, rev, inv=inv)
    rev2 = repo.get_revision(revid)
    self.assertEqual(rev, rev2)
    self.assertRaises(errors.NoSuchRevision, base.get_revision, revid)