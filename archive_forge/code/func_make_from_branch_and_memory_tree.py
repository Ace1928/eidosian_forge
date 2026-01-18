from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def make_from_branch_and_memory_tree(self, relpath):
    """Create a branch on the default transport and a MemoryTree for it."""
    self.assertEqual(self.branch_format_from._matchingcontroldir.get_branch_format(), self.branch_format_from)
    return self.make_branch_and_memory_tree(relpath, format=self.branch_format_from._matchingcontroldir)