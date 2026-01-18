import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_list_option_with_int_type_can_be_reset(self):
    options = [option.ListOption('hello', type=int)]
    opts, args = self.parse(options, ['--hello=2', '--hello=3', '--hello=-', '--hello=5'])
    self.assertEqual([5], opts.hello)