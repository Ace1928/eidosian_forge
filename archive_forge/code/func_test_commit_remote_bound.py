from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_commit_remote_bound(self):
    base_tree, child_tree = self.create_branches()
    base_tree.controldir.sprout('newbase')
    self.run_bzr('bind ../newbase', working_dir='base')
    self.run_bzr('commit -m failure --unchanged', retcode=3, working_dir='child')