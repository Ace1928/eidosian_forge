import time
from breezy import transport
from breezy.tests.per_tree import TestCaseWithTree
def get_basic_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/one'])
    tree.add(['one'])
    return self._convert_tree(tree)