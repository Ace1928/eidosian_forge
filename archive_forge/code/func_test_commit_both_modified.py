import os
from breezy import branch, builtins, errors
from breezy.tests import transport_util
def test_commit_both_modified(self):
    self.master_wt.commit('empty commit on master')
    self.start_logging_connections()
    commit = builtins.cmd_commit()
    os.chdir('local')
    self.assertRaises(errors.BoundBranchOutOfDate, commit.run, message='empty commit', unchanged=True)
    self.assertEqual(1, len(self.connections))