from breezy.builtins import cmd_branch
from breezy.tests.transport_util import TestCaseWithConnectionHookedTransport
def test_branch_remote_local(self):
    cmd = cmd_branch()
    cmd.run(self.get_url('branch'), 'local')
    self.assertEqual(1, len(self.connections))