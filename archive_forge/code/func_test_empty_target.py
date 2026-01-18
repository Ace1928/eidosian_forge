import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_empty_target(self):
    tree_a, tree_b, branch_c = self.make_trees()
    tree_d = self.make_branch_and_tree('tree_d')
    md2 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_d.branch.base, patch_type='diff', public_branch=tree_a.branch.base)