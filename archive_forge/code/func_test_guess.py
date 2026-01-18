import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_guess(self):
    commands.get_cmd_object('status')
    self.assertEqual('status', commands.guess_command('statue'))