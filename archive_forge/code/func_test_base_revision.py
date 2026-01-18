import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_base_revision(self):
    tree_a, tree_b, branch_c = self.make_trees()
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 60, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base, base_revision_id=None)
    self.assertEqual(b'rev1', md.base_revision_id)
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 60, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base, base_revision_id=b'null:')
    self.assertEqual(b'null:', md.base_revision_id)
    lines = md.to_lines()
    md2 = merge_directive.MergeDirective.from_lines(lines)
    self.assertEqual(md2.base_revision_id, md.base_revision_id)