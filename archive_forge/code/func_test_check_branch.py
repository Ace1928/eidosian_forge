from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_branch(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('foo')
    out, err = self.run_bzr('check --branch')
    self.assertContainsRe(err, "^Checking branch at '.*'\\.\\nchecked branch.*")