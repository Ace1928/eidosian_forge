import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_fires_on_all_command_names(self):
    hook_calls = []
    commands.install_bzr_command_hooks()

    def list_my_commands(cmd_names):
        hook_calls.append('called')
        cmd_names.update(['foo', 'bar'])
        return cmd_names
    commands.Command.hooks.install_named_hook('list_commands', list_my_commands, None)
    cmd = commands.get_cmd_object('info')
    self.assertEqual([], hook_calls)
    cmds = list(commands.all_command_names())
    self.assertEqual(['called'], hook_calls)
    self.assertSubset(['foo', 'bar'], cmds)