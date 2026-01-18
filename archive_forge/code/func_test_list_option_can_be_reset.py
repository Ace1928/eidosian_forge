import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_list_option_can_be_reset(self):
    """Passing an option of '-' to a list option should reset the list."""
    options = [option.ListOption('hello', type=str)]
    opts, args = self.parse(options, ['--hello=a', '--hello=b', '--hello=-', '--hello=c'])
    self.assertEqual(['c'], opts.hello)