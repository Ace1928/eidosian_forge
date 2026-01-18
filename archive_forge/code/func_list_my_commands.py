import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def list_my_commands(cmd_names):
    hook_calls.append('called')
    cmd_names.update(['foo', 'bar'])
    return cmd_names