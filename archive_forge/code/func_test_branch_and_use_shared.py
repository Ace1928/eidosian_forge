from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_branch_and_use_shared(self):
    self.run_script('$ brz init -q branch\n$ echo foo > branch/foo\n$ brz add -q branch/foo\n$ brz commit -q -m msg branch\n$ brz init-shared-repo -q .\n$ brz reconfigure --branch --use-shared branch\n$ brz info branch\nRepository branch (format: ...)\nLocation:\n  shared repository: .\n  repository branch: branch\n')