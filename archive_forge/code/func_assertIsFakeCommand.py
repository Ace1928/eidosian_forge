import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def assertIsFakeCommand(self, cmd_obj):
    from breezy.tests.fake_command import cmd_fake
    self.assertIsInstance(cmd_obj, cmd_fake)