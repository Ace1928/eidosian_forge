import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def make_minimal_branch(self, path='.', format=None):
    tree = self.make_branch_and_tree(path, format=format)
    self.build_tree([path + '/hello.txt'])
    tree.add('hello.txt')
    tree.commit(message='message1')
    return tree