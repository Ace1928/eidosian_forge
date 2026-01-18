import os
from breezy import branch, builtins, errors
from breezy.tests import transport_util
def test_commit_local(self):
    """Commits with --local should not connect to the master!"""
    self.start_logging_connections()
    commit = builtins.cmd_commit()
    os.chdir('local')
    commit.run(message='empty commit', unchanged=True, local=True)
    self.assertEqual(0, len(self.connections))