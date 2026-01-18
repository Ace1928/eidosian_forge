import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def make_linear_branch(self, path='.', format=None):
    tree = self.make_branch_and_tree(path, format=format)
    self.build_tree([path + '/hello.txt', path + '/goodbye.txt', path + '/meep.txt'])
    tree.add('hello.txt')
    tree.commit(message='message1')
    tree.add('goodbye.txt')
    tree.commit(message='message2')
    tree.add('meep.txt')
    tree.commit(message='message3')
    return tree