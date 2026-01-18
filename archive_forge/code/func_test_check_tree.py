from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_tree(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('foo')
    out, err = self.run_bzr('check --tree')
    self.assertContainsRe(err, "^Checking working tree at '.*'\\.\\n$")