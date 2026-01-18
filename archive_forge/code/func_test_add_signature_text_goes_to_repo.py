from breezy import errors
from breezy.repository import WriteGroup
from breezy.tests.per_repository_reference import \
def test_add_signature_text_goes_to_repo(self):
    tree = self.make_branch_and_tree('sample')
    revid = tree.commit('one')
    inv = tree.branch.repository.get_inventory(revid)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    base = self.make_repository('base')
    repo = self.make_referring('referring', base)
    with repo.lock_write(), WriteGroup(repo):
        rev = tree.branch.repository.get_revision(revid)
        repo.texts.add_lines((inv.root.file_id, revid), [], [])
        repo.add_revision(revid, rev, inv=inv)
        repo.add_signature_text(revid, b'text')
    repo.get_signature_text(revid)
    self.assertRaises(errors.NoSuchRevision, base.get_signature_text, revid)