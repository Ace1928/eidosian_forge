from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def make_to_branch(self, relpath):
    self.assertEqual(self.branch_format_to._matchingcontroldir.get_branch_format(), self.branch_format_to)
    return self.make_branch(relpath, format=self.branch_format_to._matchingcontroldir)