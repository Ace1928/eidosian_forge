from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_make_without_trees(self):
    repo = self.make_repository('repo', shared=True)
    repo.set_make_working_trees(True)
    self.run_bzr('reconfigure --with-no-trees', working_dir='repo')
    self.assertIs(False, repo.make_working_trees())