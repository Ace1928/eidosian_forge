from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd
import shlex
from typing import List, Optional
from absl import flags
from pyglib import appcommands
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
def do_help(self, command_name: str):
    """Print the help for command_name (if present) or general help."""

    def FormatOneCmd(name, command, command_names):
        indent_size = appcommands.GetMaxCommandLength() + 3
        if len(command_names) > 1:
            indent = ' ' * indent_size
            command_help = flags.text_wrap(command.CommandGetHelp('', cmd_names=command_names), indent=indent, firstline_indent='')
            first_help_line, _, rest = command_help.partition('\n')
            first_line = '%-*s%s' % (indent_size, name + ':', first_help_line)
            return '\n'.join((first_line, rest))
        else:
            default_indent = '  '
            return '\n' + flags.text_wrap(command.CommandGetHelp('', cmd_names=command_names), indent=default_indent, firstline_indent=default_indent) + '\n'
    if not command_name:
        print('\nHelp for Bigquery commands:\n')
        command_names = list(self._commands)
        print('\n\n'.join((FormatOneCmd(name, command, command_names) for name, command in self._commands.items() if name not in self._special_command_names)))
        print()
    elif command_name in self._commands:
        print(FormatOneCmd(command_name, self._commands[command_name], command_names=[command_name]))
    return 0