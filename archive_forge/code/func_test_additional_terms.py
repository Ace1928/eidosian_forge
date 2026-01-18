import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_additional_terms(self):
    """Additional terms can be supplied and are deduped and sorted."""
    command = self._get_command_with_see_also(['foo', 'bar'])
    self.assertEqual(['bar', 'foo', 'gam'], command.get_see_also(['gam', 'bar', 'gam']))