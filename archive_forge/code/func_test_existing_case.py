from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_existing_case(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/b'])
    tree.add(['b'])
    self.assertEqual('b', get_canonical_path(tree, 'b', lambda x: x.lower()))
    self.assertEqual('b', get_canonical_path(tree, 'B', lambda x: x.lower()))