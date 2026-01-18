import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def pre_command(cmd):
    hook_calls.append('pre')
    raise errors.CommandError()