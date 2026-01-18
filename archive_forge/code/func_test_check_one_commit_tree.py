from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_one_commit_tree(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('hallelujah')
    out, err = self.run_bzr('check')
    self.assertContainsRe(err, "Checking working tree at '.*'\\.\\n")
    self.assertContainsRe(err, "Checking repository at '.*'\\.\\n")
    self.assertContainsRe(err, 'checked repository.*\\n     1 revisions\\n     [01] file-ids\\n')
    self.assertContainsRe(err, "Checking branch at '.*'\\.\\n")
    self.assertContainsRe(err, 'checked branch.*')