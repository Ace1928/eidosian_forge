from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_initial_tree(self):
    self.make_branch_and_tree('.')
    self.run_bzr('check')