import os
from breezy import bedding, tests, workingtree
def test_logformat_gnu_changelog(self):
    wt = self.make_branch_and_tree('.')
    wt.commit('first revision', timestamp=1236045060, timezone=0)
    log, err = self.run_bzr(['log', '--log-format', 'gnu-changelog', '--timezone=utc'])
    self.assertEqual('', err)
    expected = '2009-03-03  Joe Foo  <joe@foo.com>\n\n\tfirst revision\n\n'
    self.assertEqualDiff(expected, log)