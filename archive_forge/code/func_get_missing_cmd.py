import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def get_missing_cmd(cmd_name):
    self.hook_calls.append(('called', cmd_name))
    if cmd_name in ('foo', 'info'):
        return ACommand()