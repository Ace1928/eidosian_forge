import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def make_creator(self, tree):
    creator = shelf.ShelfCreator(tree, tree.basis_tree(), [])
    self.addCleanup(creator.finalize)
    return creator