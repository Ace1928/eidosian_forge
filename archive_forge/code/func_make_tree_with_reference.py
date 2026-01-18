from breezy import controldir
from breezy.tests import TestCaseWithTransport
def make_tree_with_reference(self):
    tree = self.make_branch_and_tree('tree')
    subtree = self.make_branch_and_tree('tree/newpath')
    tree.add_reference(subtree)
    tree.set_reference_info('newpath', 'http://example.org')
    tree.commit('add reference')
    return tree