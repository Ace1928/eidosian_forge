import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_default_subclass_no_see_also(self):
    command = self._get_command_with_see_also([])
    self.assertEqual([], command.get_see_also())