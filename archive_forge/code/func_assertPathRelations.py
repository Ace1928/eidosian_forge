import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def assertPathRelations(self, previous_tree, tree, relations):
    self.assertThat(tree, HasPathRelations(previous_tree, relations))