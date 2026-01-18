from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def sprout_to(self, origdir, to_url):
    """Sprout a bzrdir, using to_format for the new branch."""
    wt = self._sprout(origdir, to_url, self.branch_format_to._matchingcontroldir)
    self.assertEqual(wt.branch._format, self.branch_format_to)
    return wt.controldir