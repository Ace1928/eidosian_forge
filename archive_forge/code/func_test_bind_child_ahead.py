from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_child_ahead(self):
    child_tree = self.create_branches()[1]
    self.run_bzr('unbind', working_dir='child')
    child_tree = child_tree.controldir.open_workingtree()
    child_tree.commit(message='child', allow_pointless=True)
    self.check_revno(2, 'child')
    self.check_revno(1, 'base')
    self.run_bzr('bind ../base', working_dir='child')
    self.check_revno(1, 'base')
    self.run_bzr('unbind', working_dir='child')
    child_tree.commit(message='child 3', allow_pointless=True)
    child_tree.commit(message='child 4', allow_pointless=True)
    child_tree.commit(message='child 5', allow_pointless=True)
    self.check_revno(5, 'child')
    self.check_revno(1, 'base')
    self.run_bzr('bind ../base', working_dir='child')
    self.check_revno(1, 'base')