import os
from breezy import bedding, tests, workingtree
def test_logformat_line_wide(self):
    """Author field should get larger for column widths over 80"""
    wt = self.make_branch_and_tree('.')
    wt.commit('revision with a long author', committer='Person with long name SENTINEL')
    log, err = self.run_bzr('log --line')
    self.assertNotContainsString(log, 'SENTINEL')
    self.overrideEnv('BRZ_COLUMNS', '116')
    log, err = self.run_bzr('log --line')
    self.assertContainsString(log, 'SENT...')
    self.overrideEnv('BRZ_COLUMNS', '0')
    log, err = self.run_bzr('log --line')
    self.assertContainsString(log, 'SENTINEL')