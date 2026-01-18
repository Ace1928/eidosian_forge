from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_make_without_trees_already_no_trees(self):
    repo = self.make_repository('repo', shared=True)
    repo.set_make_working_trees(False)
    self.run_bzr_error([" already doesn't create working trees"], 'reconfigure --with-no-trees repo')