import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_list_option_with_dash(self):
    options = [option.ListOption('with-dash', type=str)]
    opts, args = self.parse(options, ['--with-dash=world', '--with-dash=sailor'])
    self.assertEqual(['world', 'sailor'], opts.with_dash)