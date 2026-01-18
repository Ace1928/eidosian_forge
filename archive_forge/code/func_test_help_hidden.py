import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_help_hidden(self):
    c = self.get_command([option.Option('foo', hidden=True)])
    self.assertNotContainsRe(c.get_help_text(), '--foo')