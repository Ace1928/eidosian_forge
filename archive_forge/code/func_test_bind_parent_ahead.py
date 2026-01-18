from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_parent_ahead(self):
    base_tree = self.create_branches()[0]
    self.run_bzr('unbind', working_dir='child')
    base_tree.commit(message='base', allow_pointless=True)
    self.check_revno(1, 'child')
    self.run_bzr('bind ../base', working_dir='child')
    self.check_revno(1, 'child')
    self.run_bzr('unbind', working_dir='child')
    base_tree.commit(message='base 3', allow_pointless=True)
    base_tree.commit(message='base 4', allow_pointless=True)
    base_tree.commit(message='base 5', allow_pointless=True)
    self.check_revno(5, 'base')
    self.check_revno(1, 'child')
    self.run_bzr('bind ../base', working_dir='child')
    self.check_revno(1, 'child')