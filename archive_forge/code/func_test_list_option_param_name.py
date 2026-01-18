import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_list_option_param_name(self):
    """Test list options can have their param_name set."""
    options = [option.ListOption('hello', type=str, param_name='greeting')]
    opts, args = self.parse(options, ['--hello=world', '--hello=sailor'])
    self.assertEqual(['world', 'sailor'], opts.greeting)