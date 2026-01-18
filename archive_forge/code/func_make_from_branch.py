from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def make_from_branch(self, relpath):
    return self.make_branch(relpath, format=self.branch_format_from._matchingcontroldir)