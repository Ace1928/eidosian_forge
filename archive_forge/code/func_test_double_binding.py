from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_double_binding(self):
    child_tree = self.create_branches()[1]
    child_tree.controldir.sprout('child2')
    self.run_bzr('bind ../child', working_dir='child2')
    child2_tree = controldir.ControlDir.open('child2').open_workingtree()
    self.assertRaises(errors.CommitToDoubleBoundBranch, child2_tree.commit, message='child2', allow_pointless=True)