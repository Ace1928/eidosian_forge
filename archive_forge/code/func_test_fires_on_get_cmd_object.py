import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_fires_on_get_cmd_object(self):
    self.hook_missing()
    self.cmd = self.ACommand()
    self.assertEqual([], self.hook_calls)
    cmd = commands.get_cmd_object('foo')
    self.assertEqual([('called', 'foo')], self.hook_calls)
    self.assertIsInstance(cmd, self.ACommand)
    del self.hook_calls[:]
    commands.install_bzr_command_hooks()
    cmd = commands.get_cmd_object('info')
    self.assertNotEqual(None, cmd)
    self.assertEqual(0, len(self.hook_calls))