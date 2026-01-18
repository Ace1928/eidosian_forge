import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_all_commands_have_help(self):
    commands._register_builtin_commands()
    commands_without_help = set()
    base_doc = inspect.getdoc(commands.Command)
    for cmd_name in commands.all_command_names():
        cmd = commands.get_cmd_object(cmd_name)
        cmd_help = cmd.help()
        if not cmd_help or cmd_help == base_doc:
            commands_without_help.append(cmd_name)
    self.assertLength(0, commands_without_help)