from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bound_fail(self):
    base_tree, child_tree = self.create_branches()
    self.build_tree_contents([('base/a', b'new base contents\n'), ('child/b', b'new b child contents\n')])
    base_tree.commit(message='base')
    self.check_revno(2, 'base')
    self.check_revno(1, 'child')
    self.assertRaises(errors.BoundBranchOutOfDate, child_tree.commit, message='child')
    self.check_revno(1, 'child')
    child_tree.update()
    self.check_revno(2, 'child')
    child_tree.commit(message='child')
    self.check_revno(3, 'child')
    self.check_revno(3, 'base')