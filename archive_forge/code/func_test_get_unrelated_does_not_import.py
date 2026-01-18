import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_get_unrelated_does_not_import(self):
    commands.plugin_cmds.register_lazy('cmd_fake', [], 'breezy.tests.fake_command')
    self.addCleanup(self.remove_fake)
    commands.get_cmd_object('status')
    self.assertFalse(lazy_command_imported)