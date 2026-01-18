import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_from_kwargs(self):
    my_option = option.RegistryOption.from_kwargs('my-option', help='test option', short='be short', be_long='go long')
    self.assertEqual(['my-option'], [x[0] for x in my_option.iter_switches()])
    my_option = option.RegistryOption.from_kwargs('my-option', help='test option', title='My option', short='be short', be_long='go long', value_switches=True)
    self.assertEqual(['my-option', 'be-long', 'short'], [x[0] for x in my_option.iter_switches()])
    self.assertEqual('test option', my_option.help)