import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test__see_also(self):
    """When _see_also is defined, it sets the result of get_see_also()."""
    command = self._get_command_with_see_also(['bar', 'foo'])
    self.assertEqual(['bar', 'foo'], command.get_see_also())