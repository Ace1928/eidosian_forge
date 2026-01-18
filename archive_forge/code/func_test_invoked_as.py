import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_invoked_as(self):
    """The command object knows the actual name used to invoke it."""
    commands.install_bzr_command_hooks()
    commands._register_builtin_commands()
    c = commands.get_cmd_object('ci')
    self.assertIsInstance(c, builtins.cmd_commit)
    self.assertEqual(c.invoked_as, 'ci')