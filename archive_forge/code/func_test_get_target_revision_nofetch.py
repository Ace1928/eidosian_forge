import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_get_target_revision_nofetch(self):
    tree_a, tree_b, branch_c = self.make_trees()
    tree_b.branch.fetch(tree_a.branch)
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 36, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
    md.source_branch = '/dev/null'
    revision = md.install_revisions(tree_b.branch.repository)
    self.assertEqual(b'rev2a', revision)