import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
@staticmethod
def remove_fake():
    commands.plugin_cmds.remove('fake')