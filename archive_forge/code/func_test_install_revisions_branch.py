import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_install_revisions_branch(self):
    tree_a, tree_b, branch_c = self.make_trees()
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
    self.assertFalse(tree_b.branch.repository.has_revision(b'rev2a'))
    revision = md.install_revisions(tree_b.branch.repository)
    self.assertEqual(b'rev2a', revision)
    self.assertTrue(tree_b.branch.repository.has_revision(b'rev2a'))