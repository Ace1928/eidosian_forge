import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_legacy_overrides_new(self):
    mgr = utils.TestCommandManager(None)
    mgr.add_command('cmd1', FauxCommand)
    mgr.add_command('cmd2', FauxCommand2)
    mgr.add_legacy_command('cmd2', 'cmd1')
    cmd, name, remaining = mgr.find_command(['cmd2'])
    self.assertIs(cmd, FauxCommand)
    self.assertEqual(name, 'cmd2')