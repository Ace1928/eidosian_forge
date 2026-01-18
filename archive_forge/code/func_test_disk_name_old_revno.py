import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_disk_name_old_revno(self):
    tree_a, tree_b, branch_c = self.make_trees()
    tree_a.branch.nick = 'fancy-name'
    md = self.from_objects(tree_a.branch.repository, b'rev1', 500, 120, tree_b.branch.base)
    self.assertEqual('fancy-name-1', md.get_disk_name(tree_a.branch))