from io import BytesIO
from ... import errors, tests, ui
from . import TestCaseWithBranch
def test_check_branch_report_results(self):
    """Checking a branch produces results which can be printed"""
    branch = self.make_branch('.')
    branch.lock_read()
    self.addCleanup(branch.unlock)
    result = branch.check(self.make_refs(branch))
    result.report_results(verbose=True)
    result.report_results(verbose=False)